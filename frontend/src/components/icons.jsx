/* Inline SVG icon set — no icon dependency needed. */

function base({ size = 16, children, ...rest }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconCompass = (p) => base({ ...p, children: <><circle cx="12" cy="12" r="10" /><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" /></> });
export const IconMissions = (p) => base({ ...p, children: <><rect x="3" y="4" width="18" height="18" rx="2" /><path d="M16 2v4M8 2v4M3 10h18" /></> });
export const IconLayers = (p) => base({ ...p, children: <><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></> });
export const IconAgents = (p) => base({ ...p, children: <><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" /></> });
export const IconFlask = (p) => base({ ...p, children: <><path d="M9 3h6M10 3v6L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9V3" /><path d="M7 15h10" /></> });
export const IconSearch = (p) => base({ ...p, children: <><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></> });
export const IconChart = (p) => base({ ...p, children: <><path d="M3 3v18h18" /><path d="M7 15v3M12 10v8M17 6v12" /></> });
export const IconShield = (p) => base({ ...p, children: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></> });
export const IconShieldCheck = (p) => base({ ...p, children: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="m9 12 2 2 4-4" /></> });
export const IconDoc = (p) => base({ ...p, children: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><path d="M16 13H8M16 17H8" /></> });
export const IconCheck = (p) => base({ ...p, children: <path d="M20 6 9 17l-5-5" /> });
export const IconCheckCircle = (p) => base({ ...p, children: <><circle cx="12" cy="12" r="10" /><path d="m9 12 2 2 4-4" /></> });
export const IconAlert = (p) => base({ ...p, children: <><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" /><path d="M12 9v4M12 17h.01" /></> });
export const IconTarget = (p) => base({ ...p, children: <><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></> });
export const IconCoins = (p) => base({ ...p, children: <><circle cx="8" cy="8" r="6" /><path d="M18.09 10.37A6 6 0 1 1 10.34 18M7 6h1v4M16.71 13.88l.7.71-2.82 2.82" /></> });
export const IconUser = (p) => base({ ...p, children: <><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></> });
export const IconArrowRight = (p) => base({ ...p, children: <><path d="M5 12h14M12 5l7 7-7 7" /></> });
export const IconChevronDown = (p) => base({ ...p, children: <path d="m6 9 6 6 6-6" /> });
export const IconChevronUp = (p) => base({ ...p, children: <path d="m18 15-6-6-6 6" /> });
export const IconChevronLeft = (p) => base({ ...p, children: <path d="m15 18-6-6 6-6" /> });
export const IconRefresh = (p) => base({ ...p, children: <><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /><path d="M21 3v5h-5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /><path d="M8 16H3v5" /></> });
export const IconMenu = (p) => base({ ...p, children: <><path d="M4 6h16M4 12h16M4 18h16" /></> });
export const IconX = (p) => base({ ...p, children: <><path d="M18 6 6 18M6 6l12 12" /></> });
export const IconPlus = (p) => base({ ...p, children: <><path d="M12 5v14M5 12h14" /></> });
export const IconSend = (p) => base({ ...p, children: <><path d="m22 2-7 20-4-9-9-4Z" /><path d="M22 2 11 13" /></> });
export const IconStop = (p) => base({ ...p, children: <rect x="6" y="6" width="12" height="12" rx="2" /> });
export const IconClock = (p) => base({ ...p, children: <><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></> });
export const IconLink = (p) => base({ ...p, children: <><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" /><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" /></> });
export const IconSpark = (p) => base({ ...p, children: <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" /> });
export const IconCopy = (p) => base({ ...p, children: <><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></> });
export const IconSpeaker = (p) => base({ ...p, children: <><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><path d="M15.54 8.46a5 5 0 0 1 0 7.07" /></> });
export const IconThumbUp = (p) => base({ ...p, children: <><path d="M7 10v12M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" /></> });
export const IconThumbDown = (p) => base({ ...p, children: <><path d="M17 14V2M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" /></> });
export const IconCpu = (p) => base({ ...p, children: <><rect x="4" y="4" width="16" height="16" rx="2" /><rect x="9" y="9" width="6" height="6" /><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3" /></> });
