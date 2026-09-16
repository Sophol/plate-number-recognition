# FortiGate rule: allow ANPR server → PAS cameras (RTSP)

Hand this to whoever administers the FortiGate. Two values only they can supply
are marked `<...>`: the LAN/WAN interface names and the existing outbound policy
id.

## The problem

The ANPR server (`192.168.150.64`) needs to pull RTSP video from the PAS
cameras. There are two known camera addresses, and **both are blocked at the
FortiGate**, which is also the server's default gateway (`192.168.150.7`):

| Camera path              | Symptom from the server            | Cause                                    |
| ------------------------ | ---------------------------------- | ---------------------------------------- |
| `124.199.112.138` (public) | HTTP returns "Web Filter Violation"; RTSP reset | FortiGate web filter intercepts the flow |
| `10.101.10.150` (camera VLAN) | every port times out, ping 100% loss | no route/policy from `192.168.150.0/24` to `10.101.10.0/24` |

The server's routing table has only `192.168.150.0/24` direct and a default
route via the FortiGate, so reaching the camera VLAN also depends on the
FortiGate. Either path can be opened; the internal one is cleaner.

**A web-filter URL exemption does not help** — RTSP is not HTTP, so the web
filter cannot classify it and resets it regardless. What is needed is a
dedicated firewall policy for the flow **with UTM/security profiles turned off**,
placed above the general outbound policy.

## Preferred fix: reach the cameras on the internal VLAN

If the cameras are reachable at `10.101.10.0/24`, route the server there instead
of out to the public IP — no NAT, no public exposure, and inter-VLAN traffic
normally skips the web filter entirely. It needs a route (if the FortiGate is
the L3 gateway for both subnets it already has one) plus this policy:

```
config firewall address
    edit "ANPR-Server"
        set subnet 192.168.150.64/32
    next
    edit "PAS-Cameras-VLAN"
        set subnet 10.101.10.0/24
    next
end

config firewall service custom
    edit "RTSP-PAS-Cameras"
        set protocol TCP/UDP/SCTP
        set tcp-portrange 554 557 145 147
    next
end

config firewall policy
    edit 0
        set name "ANPR-to-PAS-Cameras-LAN"
        set srcintf "<interface for 192.168.150.0/24>"
        set dstintf "<interface for 10.101.10.0/24>"
        set srcaddr "ANPR-Server"
        set dstaddr "PAS-Cameras-VLAN"
        set schedule "always"
        set service "RTSP-PAS-Cameras"
        set action accept
        set utm-status disable
        set logtraffic all
        set comments "ANPR server pulls RTSP from PAS cameras on the camera VLAN"
    next
end
```

If the internal path is available, use this and skip the public-IP rule below.

## Fallback: reach the cameras on the public IP

Use this only if `10.101.10.0/24` is not reachable through the FortiGate. It is
the same idea but toward the internet-facing IP, so the UTM bypass matters more.

## Exact flow to permit

| Field       | Value                                               |
| ----------- | --------------------------------------------------- |
| Source      | `192.168.150.64/32` (ANPR / PAS Ubuntu server)      |
| Destination | `124.199.112.138/32` (PAS cameras, public IP)       |
| Service     | TCP **554, 557, 145, 147**                          |
| Action      | ACCEPT, no security profiles, `no-inspection` SSL   |

Ports from the camera list: 554 (standard RTSP), plus 145 / 147 / 557 (the
per-direction PAS forwards). RTSP is driven over **TCP**, so the media rides the
control connection — no separate UDP RTP ports are needed. Include all four TCP
ports; unused ones do no harm.

## FortiOS CLI

```
config firewall address
    edit "ANPR-Server"
        set subnet 192.168.150.64/32
    next
    edit "PAS-Cameras-Public"
        set subnet 124.199.112.138/32
    next
end

config firewall service custom
    edit "RTSP-PAS-Cameras"
        set protocol TCP/UDP/SCTP
        set tcp-portrange 554 557 145 147
    next
end

config firewall policy
    edit 0
        set name "ANPR-to-PAS-Cameras"
        set srcintf "<LAN-interface>"      # interface facing 192.168.150.0/24
        set dstintf "<WAN-interface>"      # interface toward 124.199.112.138
        set srcaddr "ANPR-Server"
        set dstaddr "PAS-Cameras-Public"
        set schedule "always"
        set service "RTSP-PAS-Cameras"
        set action accept
        set utm-status disable             # <-- the key line: no web filter / AV / app-control
        set ssl-ssh-profile "no-inspection"
        set logtraffic all
        set comments "ANPR server pulls RTSP from PAS cameras; bypass web filter (RTSP is not HTTP)"
    next
end
```

Then move it **above** the existing catch-all outbound policy, or FortiGate will
match the old web-filtered one first:

```
config firewall policy
    move <new-policy-id> before <existing-outbound-policy-id>
end
```

`show firewall policy` lists the ids; the two `<...-interface>` names come from
`show system interface` — the LAN side is the one carrying `192.168.150.0/24`.

## GUI equivalent

Policy & Objects → Firewall Policy → **Create New**: Incoming = LAN interface,
Outgoing = WAN interface, Source = `ANPR-Server`, Destination =
`PAS-Cameras-Public`, Service = `RTSP-PAS-Cameras`, Action = ACCEPT. **Leave
every Security Profile toggle OFF**, set SSL Inspection = `no-inspection`, enable
logging. Save, then **drag it above** the general outbound policy.

## Troubleshooting: connects but video is garbled

FortiGate has an RTSP **session helper/ALG** on TCP 554 that can rewrite RTSP
transport headers. It will not touch 145/147/557, but on 554 it might. If 554
misbehaves after the policy is in, disable it for testing:

```
config system session-helper
    # find the entry with 'set name rtsp' and 'set port 554', note its id, then:
    delete <id>
end
```

## Verification (run on the ANPR server once the rule is in)

```
ffprobe -rtsp_transport tcp -i "rtsp://admin:Pa%24%24w0rd@124.199.112.138:554/Streaming/Channels/101"
```

A working stream returns codec, resolution, and frame rate within a few seconds.
The `$` in the password must be URL-encoded as `%24`. Once a URL probes clean,
register the cameras in the ANPR console; the pipeline runner re-reads the
cameras table every 10 s and starts capturing with no service restart.
