const std = @import("std");

pub const Scheme = enum {
    auto,
    bf16,
    q8,

    pub fn name(self: Scheme) []const u8 {
        return switch (self) {
            .auto => "auto",
            .bf16 => "bf16",
            .q8 => "q8",
        };
    }
};

pub fn resolveScheme(cache_scheme: Scheme, backend_name: []const u8) Scheme {
    return switch (cache_scheme) {
        .auto => if (std.mem.eql(u8, backend_name, "bf16")) .bf16 else .q8,
        else => cache_scheme,
    };
}

pub const q8_group_size: usize = 16;

pub const Q8Layout = enum {
    token_major_legacy,
    head_major,
};

pub const default_q8_layout: Q8Layout = .head_major;
