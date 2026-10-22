//! TF_DSV41_PREFILL_PIPE: the names of a pipeline stage's own weights (pipe.zig renames a stage program's roles to them,
//! pipe_load.zig uploads them), host only. A stage reads every replicated tensor from the TP copy; its own full-width
//! copies are the TP-sharded tensors, under `prefix`.

const std = @import("std");
const named = @import("named.zig");

/// the weight prefix of a stage's full-width copies
pub const prefix = "P.";

/// A layer tensor's field (names.zig's TP splits): col (wq_b, the routed and shared experts' gate / up), row (wo_b, their
/// down), own groups (wo_a.<g>) or a per-head native (attn.sink); a suffix (".trellis", ".suh", ...) may follow.
/// Everything else is replicated.
pub fn sharded(field: []const u8) bool {
    const whole = [_][]const u8{ "attn.wq_b", "attn.wo_b", "attn.sink", "moe.w1", "moe.w2", "moe.w3" };
    for (whole) |w| if (std.mem.eql(u8, field, w) or (std.mem.startsWith(u8, field, w) and field.len > w.len and field[w.len] == '.')) return true;
    return std.mem.startsWith(u8, field, "attn.wo_a.") or std.mem.startsWith(u8, field, "moe.shared.");
}

/// "L<i>.<field>" -> the field (null: not a layer tensor's name).
pub fn fieldOf(name: []const u8) ?[]const u8 {
    if (name.len < 3 or name[0] != 'L') return null;
    const dot = std.mem.indexOfScalar(u8, name, '.') orelse return null;
    if (dot == 1) return null;
    for (name[1..dot]) |c| if (!std.ascii.isDigit(c)) return null;
    return name[dot + 1 ..];
}

/// A layer's world-1 images (named.Builder.layer at rank 0 of 1) down to the stage's own copies: each sharded tensor
/// renamed "P.<name>", every other image freed.
pub fn keepSharded(b: *named.Builder) !void {
    // the new names first (a failure leaves the images as they were), then the swap
    const names = try b.gpa.alloc(?[]u8, b.out.items.len);
    defer b.gpa.free(names);
    @memset(names, null);
    errdefer for (names) |x| if (x) |s| b.gpa.free(s);
    for (b.out.items, names) |t, *x| {
        const f = fieldOf(t.name) orelse continue;
        if (sharded(f)) x.* = try std.fmt.allocPrint(b.gpa, prefix ++ "{s}", .{t.name});
    }
    var kept: usize = 0;
    for (b.out.items, names) |t, x| {
        b.gpa.free(t.name);
        const nm = x orelse {
            b.gpa.free(t.bytes);
            continue;
        };
        var y = t;
        y.name = nm;
        b.out.items[kept] = y;
        kept += 1;
    }
    b.out.shrinkRetainingCapacity(kept);
}
