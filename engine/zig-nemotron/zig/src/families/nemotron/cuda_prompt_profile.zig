//! TF_NEMOTRON_PF_PROFILE=1: a prompt pass's GPU time by kernel class (cuda_forward.zig's marks), logged at its end.
//! Each chunk records an event at every class boundary, then waits for them (the pass runs one chunk at a time while
//! it profiles, so its total is a little above an unprofiled pass's). A collective's span (`comm`) holds the wait for
//! the slowest rank too.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_forward.zig");

const Class = fwd.Class;
const n_class = @typeInfo(Class).@"enum".field_names.len;
const max_marks = 4096;

pub const Profile = struct {
    ev: []cuda.Event,
    class: []Class,
    marks: fwd.Marks,
    ms: [n_class]f64 = @splat(0),
    chunks: usize = 0,
    rows: usize = 0,
    t0: std.Io.Timestamp = undefined,

    pub fn on() bool {
        const v = std.c.getenv("TF_NEMOTRON_PF_PROFILE") orelse return false;
        return !std.mem.eql(u8, std.mem.span(v), "0");
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Profile {
        const p = try gpa.create(Profile);
        errdefer gpa.destroy(p);
        const ev = try gpa.alloc(cuda.Event, max_marks);
        errdefer gpa.free(ev);
        var made: usize = 0;
        errdefer for (ev[0..made]) |*x| x.deinit();
        while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(d, true);
        const class = try gpa.alloc(Class, max_marks);
        p.* = .{ .ev = ev, .class = class, .marks = .{ .ev = ev, .class = class } };
        return p;
    }

    pub fn deinit(p: *Profile, gpa: std.mem.Allocator) void {
        for (p.ev) |*x| x.deinit();
        gpa.free(p.ev);
        gpa.free(p.class);
        gpa.destroy(p);
    }

    /// The chunk's first event (its class is never read): spans run from it.
    pub fn begin(p: *Profile, s: cuda.Stream) !void {
        p.marks.n = 0;
        try p.ev[0].record(s);
        p.class[0] = .embed;
        p.marks.n = 1;
    }

    /// After a chunk's kernels are queued: waits for them and adds each span to its class.
    pub fn end(p: *Profile, rows: usize) !void {
        const m = &p.marks;
        if (m.n < 2) return;
        try m.ev[m.n - 1].synchronize();
        for (1..m.n) |i| p.ms[@intFromEnum(m.class[i])] += try cuda.Event.elapsedMs(m.ev[i - 1], m.ev[i]);
        p.chunks += 1;
        p.rows += rows;
    }

    /// The pass's classes, largest first, as one log line.
    pub fn report(p: *const Profile, wall_s: f64) void {
        var total: f64 = 0;
        for (p.ms) |v| total += v;
        var order: [n_class]usize = undefined;
        for (&order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, &order, &p.ms, struct {
            fn lt(ms: *const [n_class]f64, a: usize, b: usize) bool {
                return ms[a] > ms[b];
            }
        }.lt);
        var buf: [2048]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        for (order) |i| {
            if (p.ms[i] < 0.5) continue;
            w.print(" {t}={d:.0}ms({d:.1}%)", .{ @as(Class, @enumFromInt(i)), p.ms[i], 100 * p.ms[i] / @max(total, 1e-9) }) catch break;
        }
        std.log.info("prompt profile: {d} rows in {d} chunks, gpu {d:.3}s wall {d:.3}s ({d:.0} tok/s):{s}", .{ p.rows, p.chunks, total / 1000, wall_s, @as(f64, @floatFromInt(p.rows)) / @max(wall_s, 1e-9), w.buffered() });
    }
};
