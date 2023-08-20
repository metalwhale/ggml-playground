const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn downloadFile(allocator: Allocator, uri_text: []const u8, save_path: []const u8) !void {
    if (try checkIfFileExists(allocator, save_path)) {
        return;
    }
    const uri = try std.Uri.parse(uri_text);
    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();
    var request = try client.request(.GET, uri, .{ .allocator = allocator }, .{});
    defer request.deinit();
    try request.start();
    try request.wait();
    var file = try std.fs.cwd().createFile(save_path, .{ .truncate = true });
    defer file.close();
    while (true) {
        var buffer: [1024 * 1024]u8 = undefined;
        const n_bytes = try request.reader().read(&buffer);
        if (n_bytes == 0) {
            break;
        }
        _ = try file.write(buffer[0..n_bytes]);
    }
}

fn checkIfFileExists(allocator: Allocator, save_path: []const u8) !bool {
    if (std.fs.realpathAlloc(allocator, save_path)) |real_save_path| {
        defer allocator.free(real_save_path);
        return true;
    } else |err| switch (err) {
        error.FileNotFound => return false,
        else => |other_err| return other_err,
    }
}
