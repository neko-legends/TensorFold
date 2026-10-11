//! TF_DSV41_PREFILL_PIPE: pipelined prompt reading on the TP ranks (docs/ZIG-PREFILL-PIPE.md). Not wired yet: the
//! rules' surface for pipe_test.zig.

const std = @import("std");
const calls = @import("calls.zig");
const block = @import("block.zig");
const Config = @import("config.zig").Config;

pub const Error = error{ BadKnob, BadBounds, NotImplemented };
pub const Get = *const fn ([]const u8) ?[]const u8;

pub const max_world: usize = 16;
pub const min_seg: u32 = 33;

pub const Settings = struct {
    min_rows: u32 = 0,
    rows: u32 = 0,
    world: u32 = 0,
    bounds: [max_world + 1]u32 = @splat(0),
    trace: bool = false,

    pub fn boundsSlice(s: *const Settings) []const u32 {
        return s.bounds[0 .. s.world + 1];
    }

    pub fn stage(s: *const Settings, r: u32) Stage {
        _ = s;
        _ = r;
        return .{};
    }

    pub fn owner(s: *const Settings, L: u32) u32 {
        _ = s;
        _ = L;
        return 0;
    }
};

pub const Stage = struct {
    rank: u32 = 0,
    lo: u32 = 0,
    hi: u32 = 0,
    d: u32 = 0,
    last: bool = false,

    pub fn layers(st: Stage, buf: []u32) []const u32 {
        _ = st;
        return buf[0..0];
    }

    pub fn weightLayers(st: Stage, buf: []u32) []const u32 {
        _ = st;
        return buf[0..0];
    }

    pub fn sources(st: Stage, cfg: *const Config, buf: []u32) []const u32 {
        _ = st;
        _ = cfg;
        return buf[0..0];
    }

    pub fn selIn(st: Stage, cfg: *const Config) ?u32 {
        _ = st;
        _ = cfg;
        return null;
    }

    pub fn selOut(st: Stage, cfg: *const Config, bounds: []const u32) ?u32 {
        _ = st;
        _ = cfg;
        _ = bounds;
        return null;
    }
};

pub fn settings(get: Get, cfg: *const Config, world: u32) !?Settings {
    _ = get;
    _ = cfg;
    _ = world;
    return error.NotImplemented;
}

pub fn boundsFor(cfg: *const Config, world: u32, raw: ?[]const u8, out: []u32) ![]const u32 {
    _ = cfg;
    _ = world;
    _ = raw;
    _ = out;
    return error.NotImplemented;
}

pub fn rowsOf(ratio: u32, start: u64, n: u32) [2]u64 {
    _ = ratio;
    _ = start;
    _ = n;
    return .{ 0, 0 };
}

pub const Seg = struct { start: u64, n: u32 };

pub const Schedule = struct {
    world: u32,

    pub fn busy(s: *const Schedule) bool {
        _ = s;
        return false;
    }

    pub fn enter(s: *Schedule, seg: ?Seg) void {
        _ = s;
        _ = seg;
    }

    pub fn round(s: *const Schedule) u64 {
        _ = s;
        return 0;
    }

    pub fn segOf(s: *const Schedule, stage_: u32) ?Seg {
        _ = s;
        _ = stage_;
        return null;
    }

    pub fn clear(s: *Schedule) void {
        _ = s;
    }
};

pub const RowsLayout = struct {
    comp_bytes: [max_world]u64 = @splat(0),
    ik_bytes: [max_world]u64 = @splat(0),
    used: [max_world]u64 = @splat(0),
    stride: u64 = 0,

    pub fn of(cfg: *const Config, s: Settings) RowsLayout {
        _ = cfg;
        _ = s;
        return .{};
    }
};

pub const FinalsLayout = struct {
    used: [max_world]u64 = @splat(0),
    stride: u64 = 0,

    pub fn of(cfg: *const Config, s: Settings) FinalsLayout {
        _ = cfg;
        _ = s;
        return .{};
    }
};

pub fn sharded(field: []const u8) bool {
    _ = field;
    return false;
}

pub fn renameRole(a: std.mem.Allocator, r: calls.Role) !calls.Role {
    _ = a;
    return r;
}

pub fn stageOptions(o: block.Options, rows: u32) block.Options {
    _ = rows;
    return o;
}
