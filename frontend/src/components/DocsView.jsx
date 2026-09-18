/* MARS documentation view — mounts the standalone docs site at `#/docs`.
 * The docs themselves are a zero-build static page (`public/docs.html`);
 * the iframe isolates the docs' own `#page` hash routing from the
 * console's `#/view` hash router. Exiting via the docs' "Console" button
 * targets the top window, which the App's hashchange sync picks up. */

import { useEffect, useRef } from "react";
import { getStoredTheme } from "../theme";

export default function DocsView() {
  const frameRef = useRef(null);

  useEffect(() => {
    const push = () => {
      const frame = frameRef.current;
      if (!frame || !frame.contentWindow) return;
      const mode = document.documentElement.getAttribute("data-theme")
        || getStoredTheme();
      frame.contentWindow.postMessage({ type: "mars-theme", mode }, "*");
    };

    push();
    const observer = new MutationObserver(push);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);

  return (
    <div className="docs-view">
      <iframe
        ref={frameRef}
        className="docs-frame"
        src="/docs.html"
        title="MARS Documentation"
        onLoad={() => {
          const frame = frameRef.current;
          if (!frame || !frame.contentWindow) return;
          const mode = document.documentElement.getAttribute("data-theme")
            || getStoredTheme();
          frame.contentWindow.postMessage({ type: "mars-theme", mode }, "*");
        }}
      />
    </div>
  );
}
