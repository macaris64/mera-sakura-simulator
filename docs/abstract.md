## SAFEGUARDING THE SEARCH FOR LIFE: A DETERMINISTIC ”SANDWICH” ARCHITECTURE FOR AUTONOMOUS BIO-SIGNATURE DETECTION IN OCEAN WORLDS

### Abstract

The search for life in Ocean Worlds, such as Europa and Enceladus, represents a pivotal frontier in
astrobiology; however, these missions are hindered by a “triple threat” of intense Jovian radiation, severe
energy constraints, and Long Fat Networks (LFN) with communication latencies reaching up to an hour.
Traditional telemetry methods cannot cope with these barriers, necessitating a transition from passive
data relays to autonomous entities capable of prioritizing high-value scientific data on-board. To address
this, this paper introduces the “Sandwich Architecture,” a hierarchical software framework governed by
NASA’s Core Flight System (cFS), designed to bridge the gap between AI-driven discovery and mission-
critical safety by decoupling probabilistic inference from deterministic validation.
The architecture is deployed across a heterogeneous hardware environment, utilizing Rad-Hard High-
Performance Spaceflight Computing (HPSC) for core command and control running cFS, and SAKURA-II
AI accelerators for intensive processing. Within this ecosystem, the cFS Software Bus (SB) serves as the
secure backbone for three integrated layers: first, the Physical Operational Constraint Layer functions
as a cFS-native gatekeeper, evaluating raw sensor data against the vehicle’s immediate health, power
states, and radiation environment via cFS Health and Safety (HS) applications. This is followed by
the Astrobiology-Specific Inference Engine, a middle layer utilizing quantized Transformer-based models
optimized for SAKURA-II edge computing to scan heterogeneous sensor streams and assign a “Science
Confidence Score.” Finally, the Deterministic Validation Layer employs a Physics-Based Deterministic
Veto Mechanism; it cross-references AI findings with environmental telemetry (temperature, pressure,
and radiation) to ensure that only thermodynamically plausible detections are prioritized.
To ensure mission-critical resilience, the architecture maintains a strict Logic Isolation (Steel Wall)
through cFS Table Management, structurally isolating Housekeeping (HK) telemetry from scientific pro-
cessing layers to prevent autonomous logic from compromising spacecraft health. Data logistics are
managed through a hybrid serialization strategy—incorporating Protocol Buffers for low-latency intra-
satellite communication between the cFS SB and SAKURA-II, and CCSDS/XTCE for space-to-ground
telemetry—while maintaining robust delivery via the CCSDS Bundle Protocol (BPv7) and Licklider Trans-
mission Protocol (LTP). Preliminary proof-of-concept simulations validate the architecture’s ability to ef-
fectively veto radiation-induced false positives while elevating verified bio-signature data to high-priority
transmission queues. By grounding probabilistic AI in rigid physical laws and cFS-level safeguards, this
architecture provides a resilient blueprint for the next generation of life-seeking probes.
