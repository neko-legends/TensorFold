//! sessions_gpu.zig is the GPU side: no host test step compiles it, yet its save gate (kv/sess.zig's
//! saveAdmits, called from Sessions.save/savePrompt before the op is broadcast) is host-tested logic
//! wired there. Keep the wiring analyzable on any host: referencing the saves forces their bodies —
//! and with them the gate, the measure and the snapshot — to compile.

const std = @import("std");
const sg = @import("sessions_gpu.zig");

test "the GPU save path stays host-analyzable (the tier gate's call sites)" {
    _ = &sg.Sessions.save;
    _ = &sg.Sessions.savePrompt;
}
