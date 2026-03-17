import { useState } from "react";

export default function SectionCard({ title, children, initiallyCollapsed = false, rightMeta = null }) {
  const [collapsed, setCollapsed] = useState(initiallyCollapsed);

  return (
    <section className="card section-card">
      <header className="section-card-header">
        <div className="section-card-title-wrap">
          <h3 className="section-card-title">{title}</h3>
          {rightMeta ? <span className="section-card-meta">{rightMeta}</span> : null}
        </div>
        <button
          type="button"
          className="section-toggle"
          onClick={() => setCollapsed((prev) => !prev)}
          aria-expanded={!collapsed}
        >
          {collapsed ? "Expand" : "Collapse"}
        </button>
      </header>
      {!collapsed ? <div className="section-card-body">{children}</div> : null}
    </section>
  );
}
