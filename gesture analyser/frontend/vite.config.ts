import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Çift yığın (dual-stack): "::" hem IPv6 ([::1]) hem IPv4-mapped (127.0.0.1)
    // dinler. Chrome Windows'ta localhost'u önce IPv6 ::1'e çözdüğü için
    // yalnızca 0.0.0.0 (IPv4) bind etmek "bağlanmayı reddetti" hatası veriyordu.
    host: "::",
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true, changeOrigin: true },
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
