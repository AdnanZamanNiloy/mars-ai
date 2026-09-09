import { Component } from "react";

/* Catches render crashes anywhere inside and shows a recoverable error
 * instead of an empty black page. Includes the error text so a bug
 * report can name the exact failure. */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    if (this.props.onError) {
      try {
        this.props.onError(error, info);
      } catch {
        /* reporting must never throw */
      }
    }
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const message = error instanceof Error ? error.message : String(error);
    return (
      <div style={{ padding: 26, maxWidth: 640, margin: "40px auto" }}>
        <div className="error-box" role="alert">
          <div><strong>Something broke while rendering this view</strong></div>
          <div style={{ marginTop: 6, fontFamily: "monospace", fontSize: 12 }}>{message}</div>
          <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
            <button className="btn" onClick={() => this.setState({ error: null })}>
              Try again
            </button>
            <button className="btn" onClick={() => window.location.reload()}>
              Reload app
            </button>
          </div>
        </div>
      </div>
    );
  }
}
