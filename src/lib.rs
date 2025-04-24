#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

//! # chili. Rust port of [Spice], a low-overhead parallelization library
//!
//! A crate for very low-overhead fork-join workloads that can potentially be
//! run in parallel.
//!
//! It works best in cases where there are many small computations and where it
//! is expensive to estimate how many are left on the current branch in order
//! to stop trying to share work across threads.
//!
//! [Spice]: https://github.com/judofyr/spice
//!
//! # Examples
//!
//! ```
//! # use chili::Scope;
//! struct Node {
//!     val: u64,
//!     left: Option<Box<Node>>,
//!     right: Option<Box<Node>>,
//! }
//!
//! impl Node {
//!     pub fn tree(layers: usize) -> Self {
//!         Self {
//!             val: 1,
//!             left: (layers != 1).then(|| Box::new(Self::tree(layers - 1))),
//!             right: (layers != 1).then(|| Box::new(Self::tree(layers - 1))),
//!         }
//!     }
//! }
//!
//! fn sum(node: &Node, scope: &mut Scope<'_>) -> u64 {
//!     let (left, right) = scope.join(
//!         |s| node.left.as_deref().map(|n| sum(n, s)).unwrap_or_default(),
//!         |s| node.right.as_deref().map(|n| sum(n, s)).unwrap_or_default(),
//!     );
//!
//!     node.val + left + right
//! }
//!
//! let tree = Node::tree(10);
//!
//! assert_eq!(sum(&tree, &mut Scope::global()), 1023);
//! ```

use std::{
    collections::{btree_map::Entry, BTreeMap, HashMap},
    io,
    num::NonZero,
    ops::{Deref, DerefMut},
    panic,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Barrier, Condvar, LockResult, Mutex, OnceLock, PoisonError, Weak,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

mod job;
pub mod tlv;

use job::{Job, JobQueue, JobStack};

/// The type for a closure that gets invoked when the Chili thread pool deadlocks
type DeadlockHandler = dyn Fn() + Send + Sync;

#[derive(Debug)]
struct Heartbeat {
    is_set: Weak<AtomicBool>,
    last_heartbeat: Instant,
}

#[derive(Debug)]
struct LockContext {
    time: u64,
    unblocked_threads: usize,
    is_stopping: bool,
    shared_jobs: BTreeMap<usize, (u64, Job)>,
    heartbeats: HashMap<u64, Heartbeat>,
    heartbeat_index: u64,
}

impl LockContext {
    fn new(unblocked_threads: usize) -> Self {
        Self {
            unblocked_threads,
            time: 0,
            is_stopping: false,
            shared_jobs: BTreeMap::new(),
            heartbeats: HashMap::new(),
            heartbeat_index: 0,
        }
    }

    pub fn new_heartbeat(&mut self) -> Arc<AtomicBool> {
        let is_set = Arc::new(AtomicBool::new(true));
        let heartbeat = Heartbeat {
            is_set: Arc::downgrade(&is_set),
            last_heartbeat: Instant::now(),
        };

        let i = self.heartbeat_index;
        self.heartbeats.insert(i, heartbeat);

        self.heartbeat_index = i.checked_add(1).unwrap();

        is_set
    }

    pub fn pop_earliest_shared_job(&mut self) -> Option<Job> {
        self.shared_jobs
            .pop_first()
            .map(|(_, (_, shared_job))| shared_job)
    }
}

#[derive(Debug)]
struct Context {
    lock: Mutex<LockContext>,
    job_is_ready: Condvar,
    scope_created_from_thread_pool: Condvar,
}

/// Thread builder used to initialize thread-locals via
/// [`ThreadPoolBuilder::spawn_handler`](struct.ThreadPoolBuilder.html#method.spawn_handler).
pub struct ThreadBuilder {
    stack_size: Option<NonZero<usize>>,
    context: Arc<Context>,
    barrier: Arc<Barrier>,
    index: usize,
}

impl ThreadBuilder {
    /// Gets the index of this thread in the pool, within `0..num_threads`.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Gets the value that was specified by `ThreadPoolBuilder::stack_size()`.
    pub fn stack_size(&self) -> Option<NonZero<usize>> {
        self.stack_size
    }

    /// Executes the main loop for this thread. This will not return until the
    /// thread pool is dropped.
    pub fn run(self) {
        let _ = execute_worker(self.context, self.barrier);
    }
}

fn execute_worker(context: Arc<Context>, barrier: Arc<Barrier>) -> LockResult<()> {
    let mut first_run = true;

    let mut job_queue = JobQueue::default();
    let mut scope = Scope::new_from_worker(context.clone(), &mut job_queue);

    loop {
        let job = {
            let mut lock = context.lock.lock().unwrap();
            lock.pop_earliest_shared_job()
        };

        if let Some(job) = job {
            // SAFETY:
            // Any `Job` that was shared between threads is waited upon before
            // the `JobStack` exits scope.
            unsafe {
                job.execute(&mut scope);
            }
        }

        if first_run {
            first_run = false;
            barrier.wait();
        };

        let lock = context.lock.lock().map_err(|_| PoisonError::new(()))?;
        // TODO: acquire/release_thread
        if lock.is_stopping || context.job_is_ready.wait(lock).is_err() {
            break;
        }
    }

    Ok(())
}

fn execute_heartbeat(
    context: Arc<Context>,
    heartbeat_interval: Duration,
    num_workers: usize,
) -> Option<()> {
    loop {
        let interval_between_workers = {
            let mut lock = context
                .scope_created_from_thread_pool
                .wait_while(context.lock.lock().ok()?, |l| {
                    l.heartbeats.len() == num_workers && !l.is_stopping
                })
                .ok()?;

            if lock.is_stopping {
                break;
            }

            let now = Instant::now();
            lock.heartbeats.retain(|_, h| {
                h.is_set
                    .upgrade()
                    .inspect(|is_set| {
                        if now.duration_since(h.last_heartbeat) >= heartbeat_interval {
                            is_set.store(true, Ordering::Relaxed);
                            h.last_heartbeat = now;
                        }
                    })
                    .is_some()
            });

            heartbeat_interval.checked_div(lock.heartbeats.len() as u32)
        };

        // If there are no heartbeats (`lock.heartbeats.len()` is 0), skip
        // immediately to the next iteration of the loop to trigger the wait.
        if let Some(interval_between_workers) = interval_between_workers {
            thread::sleep(interval_between_workers);
        }
    }

    Some(())
}

#[derive(Debug)]
enum ThreadJobQueue<'s> {
    Worker(&'s mut JobQueue),
    Current(JobQueue),
}

impl Deref for ThreadJobQueue<'_> {
    type Target = JobQueue;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Worker(queue) => queue,
            Self::Current(queue) => queue,
        }
    }
}

impl DerefMut for ThreadJobQueue<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Worker(queue) => queue,
            Self::Current(queue) => queue,
        }
    }
}

/// A `Scope`d object that you can run fork-join workloads on.
///
/// # Examples
///
/// ```
/// # use chili::ThreadPool;
/// let mut tp = ThreadPool::new();
/// let mut s = tp.scope();
///
/// let mut vals = [0; 2];
/// let (left, right) = vals.split_at_mut(1);
///
/// s.join(|_| left[0] = 1, |_| right[0] = 1);
///
/// assert_eq!(vals, [1; 2]);
/// ```
#[derive(Debug)]
pub struct Scope<'s> {
    context: Arc<Context>,
    job_queue: ThreadJobQueue<'s>,
    heartbeat: Arc<AtomicBool>,
    join_count: u8,
}

impl<'s> Scope<'s> {
    /// Returns the global scope.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::Scope;
    /// let _s = Scope::global();
    /// ```
    pub fn global() -> Scope<'static> {
        ThreadPool::global().scope()
    }

    fn new_from_thread_pool(thread_pool: &'s ThreadPool) -> Self {
        let heartbeat = thread_pool.context.lock.lock().unwrap().new_heartbeat();
        thread_pool
            .context
            .scope_created_from_thread_pool
            .notify_one();

        Self {
            context: thread_pool.context.clone(),
            job_queue: ThreadJobQueue::Current(JobQueue::default()),
            heartbeat,
            join_count: 0,
        }
    }

    fn new_from_worker(context: Arc<Context>, job_queue: &'s mut JobQueue) -> Self {
        let heartbeat = context.lock.lock().unwrap().new_heartbeat();

        Self {
            context,
            job_queue: ThreadJobQueue::Worker(job_queue),
            heartbeat,
            join_count: 0,
        }
    }

    fn heartbeat_id(&self) -> usize {
        Arc::as_ptr(&self.heartbeat) as usize
    }

    fn wait_for_sent_job<T>(&mut self, job: &Job<T>) -> Option<thread::Result<T>> {
        {
            let mut lock = self.context.lock.lock().unwrap();

            if let Some((_, job)) = lock.shared_jobs.remove(&self.heartbeat_id()) {
                // SAFETY:
                // Since the `Future` has already been allocated when
                // popping from the queue, the `Job` needs manual dropping.
                unsafe {
                    job.drop();
                }

                return None;
            }
        }

        // SAFETY:
        // For this `Job` to have crossed thread borders, it must have been
        // popped from the `JobQueue` and shared.
        while !unsafe { job.poll() } {
            let job = {
                let mut lock = self.context.lock.lock().unwrap();
                lock.pop_earliest_shared_job()
            };

            if let Some(job) = job {
                // SAFETY:
                // Any `Job` that was shared between threads is waited upon
                // before the `JobStack` exits scope.
                unsafe {
                    job.execute(self);
                }
            } else {
                break;
            }
        }

        // SAFETY:
        // Any `Job` that was shared between threads is waited upon before the
        // `JobStack` exits scope.
        unsafe { job.wait() }
    }

    #[cold]
    fn heartbeat(&mut self) {
        let mut lock = self.context.lock.lock().unwrap();

        let time = lock.time;
        if let Entry::Vacant(e) = lock.shared_jobs.entry(self.heartbeat_id()) {
            if let Some(job) = self.job_queue.pop_front() {
                e.insert((time, job));

                lock.time += 1;
                self.context.job_is_ready.notify_one();
            }
        }

        self.heartbeat.store(false, Ordering::Relaxed);
    }

    fn share_job(&mut self, job: Job) {
        let mut lock = self.context.lock.lock().unwrap();

        let time = lock.time;
        if let Entry::Vacant(e) = lock.shared_jobs.entry(self.heartbeat_id()) {
            e.insert((time, job));

            lock.time += 1;
            self.context.job_is_ready.notify_one();
        }
    }

    fn join_seq<A, B, RA, RB>(&mut self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce(&mut Scope<'_>) -> RA + Send,
        B: FnOnce(&mut Scope<'_>) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let rb = b(self);
        let ra = a(self);

        (ra, rb)
    }

    fn join_heartbeat<A, B, RA, RB>(&mut self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce(&mut Scope<'_>) -> RA + Send,
        B: FnOnce(&mut Scope<'_>) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let a = move |scope: &mut Scope<'_>| {
            if scope.heartbeat.load(Ordering::Relaxed) {
                scope.heartbeat();
            }

            a(scope)
        };

        let stack = JobStack::new(a);
        let job = Job::new(&stack);

        // SAFETY:
        // `job` is alive until the end of this scope.
        unsafe {
            self.job_queue.push_back(&job);
        }

        let rb = b(self);

        if job.is_waiting() {
            self.job_queue.pop_back();

            // SAFETY:
            // Since the `job` was popped from the back of the queue, it cannot
            // take the closure out of the `JobStack` anymore.
            // `JobStack::take_once` is thus called only once.
            (unsafe { (stack.take_once())(self) }, rb)
        } else {
            let ra = match self.wait_for_sent_job(&job) {
                Some(Ok(val)) => val,
                Some(Err(e)) => panic::resume_unwind(e),
                // SAFETY:
                // Since the `job` didn't have the chance to be actually
                // sent across threads, it cannot take the closure out of the
                // `JobStack` anymore. `JobStack::take_once` is thus called
                // only once.
                None => unsafe { (stack.take_once())(self) },
            };

            (ra, rb)
        }
    }

    /// Runs `a` and `b` potentially in parallel on separate threads and
    /// returns the results.
    ///
    /// This variant skips checking for a heartbeat every 16 calls for improved
    /// performance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::Scope;
    /// let mut vals = [0; 2];
    /// let (left, right) = vals.split_at_mut(1);
    ///
    /// Scope::global().join(|_| left[0] = 1, |_| right[0] = 1);
    ///
    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn join<A, B, RA, RB>(&mut self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce(&mut Scope<'_>) -> RA + Send,
        B: FnOnce(&mut Scope<'_>) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.join_with_heartbeat_every::<64, _, _, _, _>(a, b)
    }

    /// Runs `a` and `b` potentially in parallel on separate threads and
    /// returns the results.
    ///
    /// This variant skips checking for a heartbeat every `TIMES - 1` calls for
    /// improved performance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::Scope;
    ///
    /// let mut vals = [0; 2];
    /// let (left, right) = vals.split_at_mut(1);
    ///
    /// // Skip checking 7/8 calls to join_with_heartbeat_every.
    /// Scope::global().join_with_heartbeat_every::<8, _, _, _, _>(|_| left[0] = 1, |_| right[0] = 1);
    ///
    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn join_with_heartbeat_every<const TIMES: u8, A, B, RA, RB>(
        &mut self,
        a: A,
        b: B,
    ) -> (RA, RB)
    where
        A: FnOnce(&mut Scope<'_>) -> RA + Send,
        B: FnOnce(&mut Scope<'_>) -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.join_count = self.join_count.wrapping_add(1) % TIMES;

        if self.join_count == 0 || self.job_queue.len() < 3 {
            self.join_heartbeat(a, b)
        } else {
            self.join_seq(a, b)
        }
    }

    pub fn install<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Scope<'_>) -> R + Send,
        R: Send,
    {
        // FIXME: Temporary workaround
        let (tx, rx) = mpsc::sync_channel(1);
        let f = move |scope: &mut Scope<'_>| {
            if scope.heartbeat.load(Ordering::Relaxed) {
                scope.heartbeat();
            }

            tx.send(f(scope)).unwrap();
        };

        let stack = JobStack::new(f);
        let job = Job::for_injection(&stack);

        self.share_job(job);
        rx.recv().unwrap()
    }
}

/// `ThreadPool` configuration.
pub struct Config {
    /// The number of threads or `None` to use
    /// `std::thread::available_parallelism`.
    pub thread_count: Option<NonZero<usize>>,
    /// The interval between heartbeats on any particular thread.
    pub heartbeat_interval: Duration,
    /// Worker threads' stack size in bytes
    pub stack_size: Option<NonZero<usize>>,
    /// Closure invoked on deadlock.
    pub deadlock_handler: Option<Box<DeadlockHandler>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            thread_count: None,
            heartbeat_interval: Duration::from_micros(100),
            stack_size: None,
            deadlock_handler: None,
        }
    }
}

static GLOBAL_THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// A thread pool for running fork-join workloads.
pub struct ThreadPool {
    context: Arc<Context>,
    thread_count: usize,
    worker_handles: Vec<JoinHandle<()>>,
    heartbeat_handle: Option<JoinHandle<()>>,
    deadlock_handler: Option<Box<DeadlockHandler>>,
}

impl ThreadPool {
    /// Crates a new thread pool with default `Config`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::ThreadPool;
    /// let _tp = ThreadPool::new();
    /// ```
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Creates a new thread pool with `config`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{num::NonZero, time::Duration};
    /// # use chili::{Config, ThreadPool};
    /// let _tp = ThreadPool::with_config(Config {
    ///     thread_count: Some(NonZero::new(1).unwrap()),
    ///     heartbeat_interval: Duration::from_micros(50),
    ///     stack_size: Some(NonZero::new(128 * 1024).unwrap()),
    ///     ..Config::default()
    /// });
    /// ```
    pub fn with_config(config: Config) -> Self {
        let thread_count = config
            .thread_count
            .or_else(|| thread::available_parallelism().ok())
            .map(|thread_count| thread_count.get() - 1)
            .unwrap_or_default();
        let worker_barrier = Arc::new(Barrier::new(thread_count + 1));

        let context = Arc::new(Context {
            lock: Mutex::new(LockContext::new(thread_count)),
            job_is_ready: Condvar::new(),
            scope_created_from_thread_pool: Condvar::new(),
        });

        let worker_handles = (0..thread_count)
            .map(|_| {
                let context = context.clone();
                let barrier = worker_barrier.clone();
                let mut builder = thread::Builder::new();
                if let Some(stack_size) = config.stack_size {
                    builder = builder.stack_size(stack_size.get());
                }
                builder
                    .spawn(move || {
                        let _ = execute_worker(context, barrier);
                    })
                    .expect("Spawning worker threads")
            })
            .collect();

        worker_barrier.wait();

        Self {
            thread_count,
            context: context.clone(),
            worker_handles,
            heartbeat_handle: Some(thread::spawn(move || {
                execute_heartbeat(context, config.heartbeat_interval, thread_count);
            })),
            deadlock_handler: config.deadlock_handler,
        }
    }

    pub fn scoped_global<W, F, R>(config: Config, wrapper: W, with_pool: F) -> io::Result<R>
    where
        W: Fn(ThreadBuilder) + Sync, // expected to call `run()`
        F: FnOnce() -> R,
    {
        let thread_count = config
            .thread_count
            .or_else(|| thread::available_parallelism().ok())
            .map(|thread_count| thread_count.get())
            .unwrap_or_default();
        let worker_barrier = Arc::new(Barrier::new(thread_count + 1));

        let context = Arc::new(Context {
            lock: Mutex::new(LockContext::new(thread_count)),
            job_is_ready: Condvar::new(),
            scope_created_from_thread_pool: Condvar::new(),
        });

        std::thread::scope(|scope| {
            let wrapper = &wrapper;

            for index in 0..thread_count {
                let thread = ThreadBuilder {
                    stack_size: config.stack_size,
                    context: context.clone(),
                    barrier: worker_barrier.clone(),
                    index,
                };
                let mut builder = thread::Builder::new();
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size.get());
                }
                // FIXME: cleanup spawned threads
                builder.spawn_scoped(scope, move || wrapper(thread))?;
            }

            let pool = ThreadPool {
                thread_count,
                context: context.clone(),
                // FIXME: join is handled by drop impl
                worker_handles: Vec::new(),
                heartbeat_handle: Some(thread::spawn(move || {
                    execute_heartbeat(context, config.heartbeat_interval, thread_count);
                })),
                deadlock_handler: config.deadlock_handler,
            };
            pool.set_global()
                .ok()
                .expect("Global thread pool is already set");

            Ok(with_pool())
        })
    }

    /// Sets the global thread pool to this one.
    ///
    /// The global thread pool can only be set once. Any subsequent call will
    /// return the thread pool back.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::{num::NonZero, time::Duration};
    /// # use chili::{Config, ThreadPool};
    /// ThreadPool::with_config(Config {
    ///     thread_count: Some(NonZero::new(1).unwrap()),
    ///     heartbeat_interval: Duration::from_micros(50),
    ///     stack_size: Some(NonZero::new(128 * 1024).unwrap()),
    ///     ..Config::default()
    /// })
    /// .set_global()
    /// .ok()
    /// .unwrap();
    /// ```
    pub fn set_global(self) -> Result<(), Self> {
        GLOBAL_THREAD_POOL.set(self)
    }

    /// Returns the global thread pool.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::ThreadPool;
    /// let mut s = ThreadPool::global().scope();
    ///
    /// let mut vals = [0; 2];
    /// let (left, right) = vals.split_at_mut(1);
    ///
    /// s.join(|_| left[0] = 1, |_| right[0] = 1);
    ///
    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn global() -> &'static ThreadPool {
        GLOBAL_THREAD_POOL.get_or_init(ThreadPool::new)
    }

    /// Returns a `Scope`d object that you can run fork-join workloads on.
    ///
    /// # Examples
    ///
    /// ```
    /// # use chili::ThreadPool;
    /// let mut tp = ThreadPool::new();
    /// let mut s = tp.scope();
    ///
    /// let mut vals = [0; 2];
    /// let (left, right) = vals.split_at_mut(1);
    ///
    /// s.join(|_| left[0] = 1, |_| right[0] = 1);
    ///
    /// assert_eq!(vals, [1; 2]);
    /// ```
    pub fn scope(&self) -> Scope<'_> {
        Scope::new_from_thread_pool(self)
    }

    pub fn mark_blocked(&self) {
        let mut lcx = self.context.lock.lock().unwrap();
        lcx.unblocked_threads = lcx.unblocked_threads.checked_sub(1).unwrap();
        if lcx.unblocked_threads == 0 {
            if let Some(handler) = self.deadlock_handler.as_ref() {
                handler()
            }
        }
    }

    pub fn mark_unblocked(&self) {
        let mut lcx = self.context.lock.lock().unwrap();
        lcx.unblocked_threads += 1;
        assert!(lcx.unblocked_threads <= self.thread_count);
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.context
            .lock
            .lock()
            .expect("locking failed")
            .is_stopping = true;
        self.context.job_is_ready.notify_all();
        self.context.scope_created_from_thread_pool.notify_one();

        for handle in self.worker_handles.drain(..) {
            handle.join().unwrap();
        }

        if let Some(handle) = self.heartbeat_handle.take() {
            handle.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU8;

    use super::*;

    use thread::ThreadId;

    #[test]
    fn thread_pool_stops() {
        let _tp = ThreadPool::new();
    }

    #[test]
    fn thread_pool_with_one_thread() {
        let _tp = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(1).unwrap()),
            ..Default::default()
        });
    }

    #[test]
    fn join_basic() {
        let threat_pool = ThreadPool::new();
        let mut scope = threat_pool.scope();

        let mut a = 0;
        let mut b = 0;
        scope.join(|_| a += 1, |_| b += 1);

        assert_eq!(a, 1);
        assert_eq!(b, 1);
    }

    #[test]
    fn join_long() {
        let threat_pool = ThreadPool::new();

        fn increment(s: &mut Scope, slice: &mut [u32]) {
            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let (head, tail) = slice.split_at_mut(1);

                    s.join(|_| head[0] += 1, |s| increment(s, tail));
                }
            }
        }

        let mut vals = [0; 1_024];

        increment(&mut threat_pool.scope(), &mut vals);

        assert_eq!(vals, [1; 1_024]);
    }

    #[test]
    fn join_very_long() {
        let threat_pool = ThreadPool::new();

        fn increment(s: &mut Scope, slice: &mut [u32]) {
            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let mid = slice.len() / 2;
                    let (left, right) = slice.split_at_mut(mid);

                    s.join(|s| increment(s, left), |s| increment(s, right));
                }
            }
        }

        let mut vals = vec![0; 1_024 * 1_024];

        increment(&mut threat_pool.scope(), &mut vals);

        assert_eq!(vals, vec![1; 1_024 * 1_024]);
    }

    #[test]
    fn join_wait() {
        let threat_pool = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(2).unwrap()),
            heartbeat_interval: Duration::from_micros(1),
            ..Default::default()
        });

        fn increment(s: &mut Scope, slice: &mut [u32]) {
            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let (head, tail) = slice.split_at_mut(1);

                    s.join_with_heartbeat_every::<1, _, _, _, _>(
                        |_| {
                            thread::sleep(Duration::from_micros(10));
                            head[0] += 1;
                        },
                        |s| increment(s, tail),
                    );
                }
            }
        }

        let mut vals = [0; 10];

        increment(&mut threat_pool.scope(), &mut vals);

        assert_eq!(vals, [1; 10]);
    }

    #[test]
    #[should_panic(expected = "panicked across threads")]
    fn join_panic() {
        let threat_pool = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(2).unwrap()),
            heartbeat_interval: Duration::from_micros(1),
            stack_size: None,
            deadlock_handler: None,
        });

        if let Some(thread_count) = thread::available_parallelism().ok().map(NonZero::get) {
            if thread_count == 1 {
                // Pass test artificially when only one thread is available.
                panic!("panicked across threads");
            }
        }

        fn increment(s: &mut Scope, slice: &mut [u32], id: ThreadId) -> bool {
            let mut threads_crossed = AtomicBool::new(false);

            match slice.len() {
                0 => (),
                1 => slice[0] += 1,
                _ => {
                    let (head, tail) = slice.split_at_mut(1);

                    s.join_with_heartbeat_every::<1, _, _, _, _>(
                        |_| {
                            thread::sleep(Duration::from_micros(100));

                            if thread::current().id() != id {
                                threads_crossed.store(true, Ordering::Relaxed);
                                panic!("panicked across threads");
                            }

                            head[0] += 1;
                        },
                        |s| increment(s, tail, id),
                    );
                }
            }

            *threads_crossed.get_mut()
        }

        let mut vals = [0; 10];

        let threads_crossed =
            increment(&mut threat_pool.scope(), &mut vals, thread::current().id());

        // Since there was no panic up to this point, this means that the
        // thread boundary has not been crossed.
        //
        // Check that the work was done and pass the test artificially.
        if !threads_crossed {
            assert_eq!(vals, [1; 10]);
            panic!("panicked across threads");
        }
    }

    #[test]
    fn concurrent_scopes() {
        const NUM_THREADS: u8 = 128;
        let threat_pool = ThreadPool::with_config(Config {
            thread_count: Some(NonZero::new(4).unwrap()),
            ..Default::default()
        });

        let a = AtomicU8::new(0);
        let b = AtomicU8::new(0);

        thread::scope(|s| {
            for _ in 0..NUM_THREADS {
                s.spawn(|| {
                    let mut scope = threat_pool.scope();
                    scope.join(
                        |_| a.fetch_add(1, Ordering::Relaxed),
                        |_| b.fetch_add(1, Ordering::Relaxed),
                    );
                });
            }
        });

        assert_eq!(a.load(Ordering::Relaxed), NUM_THREADS);
        assert_eq!(b.load(Ordering::Relaxed), NUM_THREADS);
    }
}
