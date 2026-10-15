//! Tensor parallelism for the Nemotron family (zig/src/fabric/tp, from the DeepSeek TP-4 tree): the `tp` module over the
//! engine's CUDA runtime, and the mailbox/RoCE kernel image (`tp_kernel.mailbox`, empty without nvcc: NCCL only).

const std = @import("std");

/// The `tp` module over `cuda`; verbs (zig/src/fabric/tp_verbs.zig) for RoCE.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp/tp.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("verbs", b.createModule(.{ .root_source_file = b.path("zig/src/fabric/tp_verbs.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return m;
}

/// mailbox.cu as a fatbin for `sm` (GB10: 121) and compute_90 PTX.
fn mailboxFatbin(b: *std.Build, nvcc: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin", "-O3", "-std=c++17", "-gencode=arch=compute_121,code=sm_121", "-gencode=arch=compute_90,code=compute_90", "-o" });
    const out = run.addOutputFileArg("tp_mailbox.fatbin");
    run.addFileArg(b.path("zig/src/fabric/tp/mailbox.cu"));
    return out;
}

/// `pub const mailbox: []const u8`, the fatbin's bytes (8-byte aligned for Module.load) or empty.
pub fn kernelModule(b: *std.Build, nvcc: ?[]const u8) *std.Build.Module {
    const wf = b.addWriteFiles();
    if (nvcc) |tool| {
        _ = wf.addCopyFile(mailboxFatbin(b, tool), "tp_mailbox.fatbin");
        return b.createModule(.{ .root_source_file = wf.add("tp_kernel.zig", "const bytes align(16) = @embedFile(\"tp_mailbox.fatbin\").*;\npub const mailbox: []const u8 = &bytes;\n") });
    }
    return b.createModule(.{ .root_source_file = wf.add("tp_kernel.zig", "pub const mailbox: []const u8 = \"\";\n") });
}
