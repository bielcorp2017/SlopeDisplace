#include "ply_writer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

void write_binary_ply(const std::filesystem::path& path,
                      const float* xyz, size_t n,
                      const uint8_t* rgb) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write PLY: " + path.string());

    // Header matching Open3D binary_little_endian format
    std::ostringstream hdr;
    hdr << "ply\n";
    hdr << "format binary_little_endian 1.0\n";
    hdr << "comment Created by Open3D\n";
    hdr << "element vertex " << n << "\n";
    hdr << "property double x\n";
    hdr << "property double y\n";
    hdr << "property double z\n";
    if (rgb) {
        hdr << "property uchar red\n";
        hdr << "property uchar green\n";
        hdr << "property uchar blue\n";
    }
    hdr << "end_header\n";

    std::string header = hdr.str();
    out.write(header.data(), header.size());

    // Write vertex data
    for (size_t i = 0; i < n; ++i) {
        // XYZ as float64 (double)
        double x = static_cast<double>(xyz[i * 3 + 0]);
        double y = static_cast<double>(xyz[i * 3 + 1]);
        double z = static_cast<double>(xyz[i * 3 + 2]);
        out.write(reinterpret_cast<const char*>(&x), 8);
        out.write(reinterpret_cast<const char*>(&y), 8);
        out.write(reinterpret_cast<const char*>(&z), 8);
        if (rgb) {
            out.write(reinterpret_cast<const char*>(&rgb[i * 3]), 3);
        }
    }
}
