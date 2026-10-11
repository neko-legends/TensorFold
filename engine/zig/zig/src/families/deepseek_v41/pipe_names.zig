//! TF_DSV41_PREFILL_PIPE: the names of a pipeline stage's own weights (pipe.zig, pipe_load.zig), host only. Not
//! written yet: the surface host_test.zig pins.

const std = @import("std");
const named = @import("named.zig");

pub const prefix = "P.";

pub fn sharded(field: []const u8) bool {
    _ = field;
    return false;
}

pub fn fieldOf(name: []const u8) ?[]const u8 {
    _ = name;
    return null;
}

pub fn keepSharded(b: *named.Builder) !void {
    _ = b;
}
