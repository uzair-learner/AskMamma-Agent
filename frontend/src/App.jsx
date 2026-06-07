import { startTransition, useDeferredValue, useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const SESSION_STORAGE_KEY = "askmamma-session-id";

const formatter = new Intl.NumberFormat("en-US");
const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

function getSessionId() {
  const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const created = crypto.randomUUID();
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, created);
  return created;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      message = payload.detail ?? payload.message ?? message;
    } catch {
      // Keep the fallback message when the payload is not JSON.
    }
    throw new Error(message);
  }

  return response.json();
}

const cannedPrompts = [
  "Which sample demo items are low in availability right now?",
  "Generate a short AskMamma operations report.",
  "What does the return policy say about unopened items?",
];

export default function App() {
  const [sessionId] = useState(getSessionId);
  const [health, setHealth] = useState({ status: "loading", environment: "..." });
  const [dashboard, setDashboard] = useState(null);
  const [products, setProducts] = useState([]);
  const [traces, setTraces] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [reportSummary, setReportSummary] = useState("");
  const [forecastSummary, setForecastSummary] = useState("");
  const [productQuery, setProductQuery] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [isLoadingProducts, setIsLoadingProducts] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isForecasting, setIsForecasting] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Welcome to AskMamma Assistant. Ask about sample inventory, availability, forecasts, reports, or documents and I’ll route it through the agent system.",
      meta: {
        selected_agent: "SupervisorAgent",
        tools_called: [],
      },
    },
  ]);
  const [error, setError] = useState("");

  const deferredQuery = useDeferredValue(productQuery);

  async function loadOverview() {
    setError("");
    const [healthData, dashboardData, traceData] = await Promise.all([
      request("/health"),
      request("/dashboard"),
      request("/agent/traces?limit=8"),
    ]);

    startTransition(() => {
      setHealth(healthData);
      setDashboard(dashboardData);
      setTraces(traceData);
    });
  }

  async function loadProducts(query = "") {
    setIsLoadingProducts(true);
    try {
      const params = new URLSearchParams();
      params.set("limit", "100");
      if (query) {
        params.set("search", query);
      }
      const items = await request(`/demo/items?${params.toString()}`);
      startTransition(() => {
        setProducts(items);
        setSelectedProduct((current) => {
          if (!items.length) {
            return null;
          }
          if (current) {
            return items.find((item) => item.id === current.id) ?? items[0];
          }
          return items[0];
        });
      });
    } finally {
      setIsLoadingProducts(false);
    }
  }

  useEffect(() => {
    Promise.all([loadOverview(), loadProducts()]).catch((loadError) => {
      setError(loadError.message);
    });
  }, []);

  useEffect(() => {
    loadProducts(deferredQuery).catch((loadError) => {
      setError(loadError.message);
    });
  }, [deferredQuery]);

  const topSignals = dashboard?.predicted_high_demand_products ?? [];
  const recentActions = dashboard?.recent_ai_actions ?? [];

  async function refreshAll() {
    setError("");
    try {
      await Promise.all([loadOverview(), loadProducts(deferredQuery)]);
    } catch (refreshError) {
      setError(refreshError.message);
    }
  }

  async function handleGenerateReport() {
    setIsGeneratingReport(true);
    setError("");
    try {
      const report = await request("/reports/askmamma");
      setReportSummary(report.summary);
      await loadOverview();
    } catch (reportError) {
      setError(reportError.message);
    } finally {
      setIsGeneratingReport(false);
    }
  }

  async function handleRunForecast() {
    setIsForecasting(true);
    setError("");
    try {
      const forecast = await request("/demo/forecast", {
        method: "POST",
        body: JSON.stringify({ months: 6 }),
      });
      setForecastSummary(forecast.explanation ?? forecast.message ?? "Forecast completed.");
      await loadOverview();
    } catch (forecastError) {
      setError(forecastError.message);
    } finally {
      setIsForecasting(false);
    }
  }

  async function submitChatMessage(message) {
    if (!message.trim()) {
      return;
    }

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: message,
    };
    setMessages((current) => [...current, userMessage]);
    setChatInput("");
    setIsSending(true);
    setError("");

    try {
      const result = await request("/agent/chat", {
        method: "POST",
        body: JSON.stringify({
          message,
          session_id: sessionId,
        }),
      });

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: result.answer,
        meta: {
          selected_agent: result.selected_agent,
          tools_called: result.tools_called ?? [],
          route_path: result.route_path ?? [],
          latency_ms: result.latency_ms,
        },
      };
      setMessages((current) => [...current, assistantMessage]);
      await loadOverview();
    } catch (chatError) {
      setError(chatError.message);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-error-${Date.now()}`,
          role: "assistant",
          content: `I hit an error while contacting the agent: ${chatError.message}`,
          meta: {
            selected_agent: "SupervisorAgent",
            tools_called: [],
          },
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  const metrics = [
    {
      label: "Sample Items",
      value: dashboard ? formatter.format(dashboard.total_products) : "--",
      accent: "blue",
    },
    {
      label: "Low Availability",
      value: dashboard ? formatter.format(dashboard.low_stock_products) : "--",
      accent: "amber",
    },
    {
      label: "Unavailable",
      value: dashboard ? formatter.format(dashboard.out_of_stock_products) : "--",
      accent: "coral",
    },
    {
      label: "Forecast Signals",
      value: dashboard ? formatter.format(topSignals.length) : "--",
      accent: "mint",
    },
  ];

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="app-frame">
        <section className="hero-card">
          <div className="hero-copy">
            <div className="eyebrow-row">
              <span className="eyebrow">AskMamma Operations Studio</span>
              <span className={`status-pill ${health.status === "ok" ? "status-ok" : "status-waiting"}`}>
                <span className="status-dot" />
                {health.status === "ok" ? `API live in ${health.environment}` : "Connecting"}
              </span>
            </div>

            <h1>Show users a sharper, faster AskMamma experience.</h1>
            <p>
              A React presentation layer for inventory signals, AI operations support, report generation, and
              document-backed answers powered by your existing FastAPI agent backend.
            </p>

            <div className="hero-actions">
              <button className="primary-button" onClick={refreshAll}>
                Refresh live data
              </button>
              <button className="secondary-button" onClick={handleGenerateReport} disabled={isGeneratingReport}>
                {isGeneratingReport ? "Generating report..." : "Generate report"}
              </button>
              <button className="ghost-button" onClick={handleRunForecast} disabled={isForecasting}>
                {isForecasting ? "Running forecast..." : "Run forecast"}
              </button>
            </div>

            {error ? <div className="error-banner">{error}</div> : null}
            {reportSummary ? <div className="info-banner">Report: {reportSummary}</div> : null}
            {forecastSummary ? <div className="info-banner">Forecast: {forecastSummary}</div> : null}
          </div>

          <div className="hero-panel">
            <div className="hero-panel-header">
              <span>Live signal board</span>
              <span>{new Date().toLocaleDateString()}</span>
            </div>
            <div className="metrics-grid">
              {metrics.map((metric) => (
                <article key={metric.label} className={`metric-card metric-${metric.accent}`}>
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                </article>
              ))}
            </div>
            <div className="signals-list">
              <div className="section-label">High-demand demo signals</div>
              {topSignals.length ? (
                topSignals.map((signal) => (
                  <div key={signal.name} className="signal-row">
                    <span>{signal.name}</span>
                    <strong>{formatter.format(signal.sold)} sold</strong>
                  </div>
                ))
              ) : (
                <div className="empty-state compact">No forecast signals yet.</div>
              )}
            </div>
          </div>
        </section>

        <section className="content-grid">
          <div className="left-rail">
            <section className="panel panel-large">
              <div className="panel-header">
                <div>
                  <p className="section-label">Catalog</p>
                  <h2>Sample inventory view</h2>
                </div>
                <div className="catalog-search">
                  <input
                    value={productQuery}
                    onChange={(event) => setProductQuery(event.target.value)}
                    placeholder="Search SKU, product, category, partner..."
                  />
                </div>
              </div>

              <div className="table-layout">
                <div className="table-card">
                  <div className="table-toolbar">
                    <span>{isLoadingProducts ? "Loading products..." : `${products.length} items loaded`}</span>
                    <span>Click a row to inspect details</span>
                  </div>

                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Item</th>
                          <th>Category</th>
                          <th>Supplier</th>
                          <th>Stock</th>
                          <th>Threshold</th>
                          <th>Price</th>
                        </tr>
                      </thead>
                      <tbody>
                        {products.map((product) => (
                          <tr
                            key={product.id}
                            className={selectedProduct?.id === product.id ? "selected-row" : ""}
                            onClick={() => setSelectedProduct(product)}
                          >
                            <td>
                              <div className="product-cell">
                                <strong>{product.name}</strong>
                                <span>{product.sku}</span>
                              </div>
                            </td>
                            <td>{product.category}</td>
                            <td>{product.supplier_name ?? "Unassigned"}</td>
                            <td>{formatter.format(product.stock_quantity)}</td>
                            <td>{formatter.format(product.reorder_level)}</td>
                            <td>{currencyFormatter.format(product.price)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {!products.length ? <div className="empty-state">No items matched your search.</div> : null}
                  </div>
                </div>

                <aside className="detail-card">
                  <div className="section-label">Selected item</div>
                  {selectedProduct ? (
                    <>
                      <h3>{selectedProduct.name}</h3>
                      <p>{selectedProduct.description || "No description provided for this sample demo item."}</p>
                      <dl className="detail-grid">
                        <Detail label="SKU" value={selectedProduct.sku} />
                        <Detail label="Category" value={selectedProduct.category} />
                        <Detail label="Supplier" value={selectedProduct.supplier_name ?? "Unassigned"} />
                        <Detail label="Location" value={selectedProduct.location ?? "N/A"} />
                        <Detail label="Stock" value={formatter.format(selectedProduct.stock_quantity)} />
                        <Detail label="Reorder Qty" value={formatter.format(selectedProduct.reorder_quantity)} />
                        <Detail label="Threshold" value={formatter.format(selectedProduct.reorder_level)} />
                        <Detail label="Price" value={currencyFormatter.format(selectedProduct.price)} />
                      </dl>
                    </>
                  ) : (
                    <div className="empty-state">Choose an item from the table to see its details here.</div>
                  )}
                </aside>
              </div>
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <p className="section-label">Traceability</p>
                  <h2>Recent agent traces</h2>
                </div>
              </div>

              <div className="trace-list">
                {traces.length ? (
                  traces.map((trace, index) => (
                    <article key={`${trace.session_id}-${index}`} className="trace-card">
                      <div className="trace-topline">
                        <span>{trace.selected_agent || "Agent"}</span>
                        <span>{new Date(trace.created_at).toLocaleString()}</span>
                      </div>
                      <p>{trace.user_input}</p>
                      <div className="trace-meta">
                        <span>Session {trace.session_id.slice(0, 8)}</span>
                      </div>
                    </article>
                  ))
                ) : (
                  <div className="empty-state">No traces yet. Ask the assistant something to generate one.</div>
                )}
              </div>
            </section>
          </div>

          <div className="right-rail">
            <section className="panel chat-panel">
              <div className="panel-header">
                <div>
                  <p className="section-label">Assistant</p>
                  <h2>AI operations chat</h2>
                </div>
                <span className="session-pill">Session {sessionId.slice(0, 8)}</span>
              </div>

              <div className="prompt-row">
                {cannedPrompts.map((prompt) => (
                  <button key={prompt} className="prompt-chip" onClick={() => submitChatMessage(prompt)} disabled={isSending}>
                    {prompt}
                  </button>
                ))}
              </div>

              <div className="chat-stream">
                {messages.map((message) => (
                  <article key={message.id} className={`message-bubble message-${message.role}`}>
                    <div className="message-role">{message.role === "assistant" ? "AskMamma AI" : "You"}</div>
                    <p>{message.content}</p>
                    {message.meta ? (
                      <div className="message-meta">
                        <span>{message.meta.selected_agent}</span>
                        {message.meta.tools_called?.length ? (
                          <span>Tools: {message.meta.tools_called.join(", ")}</span>
                        ) : null}
                        {message.meta.latency_ms ? <span>{message.meta.latency_ms} ms</span> : null}
                      </div>
                    ) : null}
                  </article>
                ))}
              </div>

              <form
                className="composer"
                onSubmit={(event) => {
                  event.preventDefault();
                  submitChatMessage(chatInput);
                }}
              >
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="Ask about availability, partners, forecasts, documents, or reports..."
                  rows={4}
                />
                <button className="primary-button" type="submit" disabled={isSending || !chatInput.trim()}>
                  {isSending ? "Sending..." : "Ask AskMamma"}
                </button>
              </form>
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <p className="section-label">Highlights</p>
                  <h2>Operational watchlist</h2>
                </div>
              </div>

              <div className="watch-grid">
                <div className="watch-card">
                  <span>Recent AI actions</span>
                  {recentActions.length ? (
                    recentActions.map((action, index) => (
                      <div key={`${action.created_at}-${index}`} className="watch-row">
                        <strong>{action.selected_agent || "Agent"}</strong>
                        <p>{action.final_answer}</p>
                      </div>
                    ))
                  ) : (
                    <div className="empty-state compact">No recent actions captured yet.</div>
                  )}
                </div>

                <div className="watch-card">
                  <span>Demo reminders</span>
                  <ul className="check-list">
                    <li>Forecasts are based on sample demo history.</li>
                    <li>Reports save to the local `outputs/reports` folder.</li>
                    <li>Document answers use the indexed local knowledge base.</li>
                  </ul>
                </div>
              </div>
            </section>
          </div>
        </section>
      </main>
    </div>
  );
}

function Detail({ label, value }) {
  return (
    <div className="detail-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
