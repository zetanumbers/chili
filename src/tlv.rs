//! Allows access to the Chili's thread local value
//! which is preserved when moving jobs across threads

use std::{cell::Cell, ptr};

thread_local! {
    /// A thread local variable that is preserved between worker thread boundaries
    pub static TLV: Cell<*const ()> = const { Cell::new(ptr::null()) }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Tlv(pub(crate) *const ());

// SAFETY: TLV is expected to be shared across threads
unsafe impl Sync for Tlv {}
// SAFETY: TLV is expected to be transferred across threads
unsafe impl Send for Tlv {}

/// Sets the current thread-local value
#[inline]
pub(crate) fn set(value: Tlv) {
    TLV.with(|tlv| tlv.set(value.0));
}

/// Returns the current thread-local value
#[inline]
pub(crate) fn get() -> Tlv {
    TLV.with(|tlv| Tlv(tlv.get()))
}
