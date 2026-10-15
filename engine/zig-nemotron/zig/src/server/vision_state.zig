//! A multimodal server's image side: the prepared-image host (decoding, fetching, the processor, virtual ids) and the
//! engine's `embed` hook. A request's images are prepared and embedded while its prompt renders; the prompt's `<image>`
//! placeholders then become `<img>` + the image's virtual ids + `</img>`, and the engine reads the rows the ids name.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const log = @import("log.zig");
const vision = @import("vision/host.zig");
const prep = @import("vision/prep.zig");
const parts = @import("vision/parts.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const Cx = errors.Cx;

pub const State = struct {
    gpa: Allocator,
    host: vision.Host,
    embedder: api.Vision,
    image: u32,
    start: u32,
    end: u32,

    /// Null (with the reason logged) when the tokenizer lacks `<image>`, `<img>` or `</img>`.
    pub fn init(gpa: Allocator, io: std.Io, embedder: api.Vision, text: model_text.Text, environ: ?*const std.process.Environ.Map) !?*State {
        const image = text.tokenId("<image>") orelse return missing("<image>");
        const start = text.tokenId("<img>") orelse return missing("<img>");
        const end = text.tokenId("</img>") orelse return missing("</img>");
        var problem: []const u8 = "";
        const settings = prep.Settings.read(environ, &problem) catch {
            log.line("vision off: bad setting ({s})", .{problem});
            return null;
        };
        const st = try gpa.create(State);
        st.* = .{ .gpa = gpa, .host = vision.Host.init(gpa, io, settings), .embedder = embedder, .image = image, .start = start, .end = end };
        try st.host.enableFetch();
        log.line("images in: <image> id {d}, up to {d} patches an image, {d} images a request", .{ image, settings.max_patches, settings.max_images });
        return st;
    }

    fn missing(name: []const u8) ?*State {
        log.line("vision off: the tokenizer has no {s} token", .{name});
        return null;
    }

    pub fn deinit(st: *State) void {
        st.host.deinit();
        st.gpa.destroy(st);
    }

    /// `ids` with each `<image>` placeholder expanded for the request's images (in prompt order, `urls`).
    pub fn expand(st: *State, cx: *Cx, ids: []const u32, urls: []const []const u8) errors.Refused![]u32 {
        var problem: parts.Problem = .{};
        const held = st.host.images(cx.a, urls, &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return if (problem.busy) cx.fail(.capacity, "{s}", .{problem.message}) else cx.refuse(problem.message),
        };
        defer st.host.releaseAll(held);
        const prepared = try cx.a.alloc(prep.Prepared, held.len);
        for (held, prepared) |h, *p| p.* = h.p;
        for (prepared) |*p| {
            const img: api.HeldImage = .{ .pixels = p.patches.ptr, .rows = p.rows, .cols = p.cols, .digest = p.digest, .vids = p.vids.ptr, .n_vids = p.vids.len };
            st.embedder.embed(st.embedder.ctx, &img) catch |e| return cx.fail(.server, "the image could not be embedded ({s})", .{@errorName(e)});
        }
        const out = parts.expand(cx.a, ids, prepared, st.image, st.start, st.end, &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return cx.refuse(problem.message),
        };
        if (std.c.getenv("TF_NEMOTRON_VISION_DUMP")) |dir| st.dump(std.mem.span(dir), out, prepared);
        return out;
    }

    /// TF_NEMOTRON_VISION_DUMP=DIR (dev): the expanded prompt ids and each image's pixels and grid, for the reference
    /// comparison (tools/nemotron/vision_e2e_ref.py); the last request's overwrite the earlier.
    fn dump(st: *State, dir: []const u8, ids: []const u32, prepared: []const prep.Prepared) void {
        const io = st.host.io;
        var buf: [512]u8 = undefined;
        const d = std.Io.Dir.cwd();
        const path = std.fmt.bufPrint(&buf, "{s}/ids.bin", .{dir}) catch return;
        d.writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(ids) }) catch return;
        for (prepared, 0..) |p, k| {
            const px = std.fmt.bufPrint(&buf, "{s}/img{d}.bin", .{ dir, k }) catch return;
            d.writeFile(io, .{ .sub_path = px, .data = std.mem.sliceAsBytes(p.patches) }) catch return;
            var meta: [128]u8 = undefined;
            const text = std.fmt.bufPrint(&meta, "{{\"rows\": {d}, \"cols\": {d}, \"vids\": {d}}}", .{ p.rows, p.cols, p.vids.len }) catch return;
            const mp = std.fmt.bufPrint(&buf, "{s}/img{d}.json", .{ dir, k }) catch return;
            d.writeFile(io, .{ .sub_path = mp, .data = text }) catch return;
        }
    }
};

/// The image URLs of normalized messages, in prompt order (each message's content parts in order).
pub fn imageUrls(a: Allocator, messages: Value) errors.Refused![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    if (messages != .array) return out.items;
    for (messages.array) |m| {
        const content = m.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array) |part| {
            if (!isImagePart(part)) continue;
            const url = parts.partUrl(a, part) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Refused => return error.Refused,
            };
            try out.append(a, url);
        }
    }
    return out.items;
}

pub fn isImagePart(part: Value) bool {
    if (part != .object) return false;
    const kind = part.strField("type") orelse return false;
    return std.mem.eql(u8, kind, "image_url") or std.mem.eql(u8, kind, "image") or std.mem.eql(u8, kind, "input_image");
}
