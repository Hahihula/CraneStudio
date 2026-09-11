//! LAN address detection for §10's "what to print when the server is up" —
//! once the gateway is bound to `0.0.0.0` (§10.1's opt-in LAN exposure),
//! another machine needs the *actual* address to connect to, not `0.0.0.0`
//! (which isn't itself a reachable destination).

use std::net::UdpSocket;

/// The local address the OS would use to reach the outside world, per the
/// standard "connect a UDP socket, read its local address back" trick — no
/// packet is actually sent (`connect` on a UDP socket only picks a route),
/// so this makes no network traffic and needs no external service.
#[must_use]
pub fn local_ip() -> Option<std::net::IpAddr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    socket.local_addr().ok().map(|addr| addr.ip())
}
