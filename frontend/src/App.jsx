import { startTransition, useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const SESSION_STORAGE_KEY = "askmamma-session-id";

const formatter = new Intl.NumberFormat("en-US");
const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const cannedPrompts = [
  "Which sample demo items are low in availability right now?",
  "Generate a short AskMamma operations report.",
  "What does the return policy say about unopened items?",
];

const navigationItems = [
  { label: "Dashboard", route: "dashboard" },
  { label: "Products", route: "products" },
  { label: "High Demand", route: "products", params: { filter: "high-demand" } },
  { label: "Forecasts", route: "forecasts" },
  { label: "Reorder", route: "reorder" },
  { label: "Suppliers", route: "suppliers" },
  { label: "Reports", route: "reports" },
  { label: "Ask AskMamma", route: "ask" },
  { label: "AI Architecture", route: "architecture" },
  { label: "Admin", route: "admin" },
];

const dashboardCardConfig = [
  {
    title: "All Products",
    route: "products",
    accent: "blue",
    description: "Browse the full sample catalog, search items, and inspect product details.",
    action: "View all products",
  },
  {
    title: "Low Stock",
    route: "products",
    params: { filter: "low-stock" },
    accent: "amber",
    description: "Items below their current threshold and likely to need attention soon.",
    action: "View low-stock items",
  },
  {
    title: "Out of Stock",
    route: "products",
    params: { filter: "out-of-stock" },
    accent: "coral",
    description: "Products currently unavailable in the sample catalog.",
    action: "View unavailable items",
  },
  {
    title: "High Demand Products",
    route: "products",
    params: { filter: "high-demand" },
    accent: "mint",
    description: "Products with strong recent sales or usage signals.",
    action: "View high-demand items",
  },
  {
    title: "Forecast Alerts",
    route: "forecasts",
    accent: "violet",
    description: "Products where historical usage suggests near-term demand pressure.",
    action: "View forecast alerts",
  },
  {
    title: "Reorder Recommendations",
    route: "reorder",
    accent: "sunset",
    description: "Suggested purchasing actions based on current stock and deterministic forecasts.",
    action: "View reorder plan",
  },
  {
    title: "Suppliers",
    route: "suppliers",
    accent: "teal",
    description: "See supplier coverage, low-stock exposure, and demo partner details.",
    action: "View suppliers",
  },
  {
    title: "Reports",
    route: "reports",
    accent: "slate",
    description: "Generate and review AskMamma operations reports and report history.",
    action: "View reports",
  },
  {
    title: "Ask AskMamma",
    route: "ask",
    accent: "indigo",
    description: "Open the AI operations assistant for inventory, forecast, or document questions.",
    action: "Open AI chat",
  },
  {
    title: "AI Architecture",
    route: "architecture",
    accent: "rose",
    description: "Understand how the supervisor and specialist agents handle business requests.",
    action: "Explore agent flow",
  },
  {
    title: "Admin / Traces",
    route: "admin",
    accent: "graphite",
    description: "Review technical traces, recent task activity, and internal operational detail.",
    action: "Open admin traces",
  },
];

function getSessionId() {
  const existing = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const created = crypto.randomUUID();
  window.sessionStorage.setItem(SESSION_STORAGE_KEY, created);
  return created;
}

function parseHash() {
  const raw = window.location.hash.replace(/^#/, "");
  if (!raw) {
    return { route: "dashboard", params: {} };
  }
  const [routePart, queryString = ""] = raw.split("?");
  const params = Object.fromEntries(new URLSearchParams(queryString).entries());
  return { route: routePart || "dashboard", params };
}

function navigateTo(route, params = {}) {
  const query = new URLSearchParams(params).toString();
  window.location.hash = query ? `${route}?${query}` : route;
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

export default function App() {
  const [sessionId] = useState(getSessionId);
  const [routeState, setRouteState] = useState(parseHash);
  const [health, setHealth] = useState({ status: "loading", environment: "..." });
  const [dashboard, setDashboard] = useState(null);
  const [products, setProducts] = useState([]);
  const [suppliers, setSuppliers] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [reports, setReports] = useState([]);
  const [traces, setTraces] = useState([]);
  const [selectedProductId, setSelectedProductId] = useState(null);
  const [productQuery, setProductQuery] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [reportSummary, setReportSummary] = useState("");
  const [forecastSummary, setForecastSummary] = useState("");
  const [error, setError] = useState("");
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Welcome to AskMamma Assistant. Ask about sample inventory, availability, forecasts, reports, or documents and I'll route it through the agent system.",
      meta: {
        selected_agent: "SupervisorAgent",
        tools_called: [],
      },
    },
  ]);

  useEffect(() => {
    const syncRoute = () => setRouteState(parseHash());
    window.addEventListener("hashchange", syncRoute);
    if (!window.location.hash) {
      navigateTo("dashboard");
    }
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  async function loadAllData() {
    const [healthData, dashboardData, productData, supplierData, recommendationData, reportData, traceData] =
      await Promise.all([
        request("/health"),
        request("/dashboard"),
        request("/demo/items?limit=100"),
        request("/demo/suppliers"),
        request("/demo/recommendations/reorder"),
        request("/reports"),
        request("/agent/traces?limit=20"),
      ]);

    startTransition(() => {
      setHealth(healthData);
      setDashboard(dashboardData);
      setProducts(productData);
      setSuppliers(supplierData);
      setRecommendations(recommendationData);
      setReports(reportData);
      setTraces(traceData);
      setSelectedProductId((current) => current ?? productData[0]?.id ?? null);
    });
  }

  useEffect(() => {
    loadAllData().catch((loadError) => {
      setError(loadError.message);
    });
  }, []);

  const highDemandProducts = useMemo(() => {
    const signalNames = new Set((dashboard?.predicted_high_demand_products ?? []).map((item) => item.name));
    return products.filter((product) => signalNames.has(product.name));
  }, [dashboard, products]);

  const forecastAlerts = useMemo(
    () => recommendations.filter((item) => Number(item.recommended_quantity) > 0),
    [recommendations],
  );

  const productCounts = useMemo(
    () => ({
      all: products.length,
      lowStock: products.filter((item) => item.stock_quantity > 0 && item.stock_quantity <= item.reorder_level).length,
      outOfStock: products.filter((item) => item.stock_quantity <= 0).length,
      highDemand: highDemandProducts.length,
      forecastAlerts: forecastAlerts.length,
      reorder: recommendations.length,
      suppliers: suppliers.length,
      reports: reports.length,
      ask: messages.length - 1,
      architecture: 5,
      admin: traces.length,
    }),
    [forecastAlerts.length, highDemandProducts.length, messages.length, products, recommendations.length, reports.length, suppliers.length, traces.length],
  );

  const activeProductFilter = routeState.params.filter ?? "all";

  const visibleProducts = useMemo(() => {
    const searchValue = productQuery.trim().toLowerCase();
    let filtered = products;
    if (activeProductFilter === "low-stock") {
      filtered = products.filter((item) => item.stock_quantity > 0 && item.stock_quantity <= item.reorder_level);
    } else if (activeProductFilter === "out-of-stock") {
      filtered = products.filter((item) => item.stock_quantity <= 0);
    } else if (activeProductFilter === "high-demand") {
      const signalNames = new Set(highDemandProducts.map((item) => item.name));
      filtered = products.filter((item) => signalNames.has(item.name));
    }

    if (!searchValue) {
      return filtered;
    }

    return filtered.filter((item) =>
      [item.name, item.sku, item.category, item.supplier_name, item.location]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(searchValue)),
    );
  }, [activeProductFilter, highDemandProducts, productQuery, products]);

  const selectedProduct =
    visibleProducts.find((item) => item.id === selectedProductId) ??
    products.find((item) => item.id === selectedProductId) ??
    visibleProducts[0] ??
    null;

  useEffect(() => {
    if (selectedProduct) {
      setSelectedProductId(selectedProduct.id);
    }
  }, [selectedProduct?.id]);

  async function refreshAll() {
    setIsRefreshing(true);
    setError("");
    try {
      await loadAllData();
    } catch (refreshError) {
      setError(refreshError.message);
    } finally {
      setIsRefreshing(false);
    }
  }

  async function handleGenerateReport() {
    setIsGeneratingReport(true);
    setError("");
    try {
      const report = await request("/reports/askmamma");
      setReportSummary(report.summary);
      await loadAllData();
      navigateTo("reports");
    } catch (reportError) {
      setError(reportError.message);
    } finally {
      setIsGeneratingReport(false);
    }
  }

  async function handleRunForecast() {
    setError("");
    try {
      const forecast = await request("/demo/forecast", {
        method: "POST",
        body: JSON.stringify({ months: 6 }),
      });
      setForecastSummary(forecast.explanation ?? forecast.message ?? "Forecast completed.");
      await loadAllData();
      navigateTo("forecasts");
    } catch (forecastError) {
      setError(forecastError.message);
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

      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: result.answer,
          meta: {
            selected_agent: result.selected_agent,
            tools_called: result.tools_called ?? [],
            latency_ms: result.latency_ms,
          },
        },
      ]);
      await loadAllData();
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

  const dashboardCards = dashboardCardConfig.map((card) => ({
    ...card,
    count:
      {
        "All Products": productCounts.all,
        "Low Stock": productCounts.lowStock,
        "Out of Stock": productCounts.outOfStock,
        "High Demand Products": productCounts.highDemand,
        "Forecast Alerts": productCounts.forecastAlerts,
        "Reorder Recommendations": productCounts.reorder,
        Suppliers: productCounts.suppliers,
        Reports: productCounts.reports,
        "Ask AskMamma": productCounts.ask,
        "AI Architecture": productCounts.architecture,
        "Admin / Traces": productCounts.admin,
      }[card.title] ?? "--",
  }));

  return (
    <div className="workspace-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <aside className="sidebar">
        <div className="brand-block">
          <span className="eyebrow">AskMamma</span>
          <h1>Operations Studio</h1>
          <p>Move by business category instead of hunting through a big table first.</p>
        </div>

        <nav className="sidebar-nav">
          {navigationItems.map((item) => {
            const isActive =
              routeState.route === item.route &&
              (!item.params?.filter || routeState.params.filter === item.params.filter);
            return (
              <button
                key={`${item.route}-${item.label}`}
                className={`nav-button ${isActive ? "nav-button-active" : ""}`}
                onClick={() => navigateTo(item.route, item.params ?? {})}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <span className={`status-pill ${health.status === "ok" ? "status-ok" : "status-waiting"}`}>
            <span className="status-dot" />
            {health.status === "ok" ? `API live in ${health.environment}` : "Connecting"}
          </span>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div>
            <p className="section-label">AskMamma Operations Studio</p>
            <h2>{pageTitle(routeState)}</h2>
          </div>
          <div className="topbar-actions">
            <button className="secondary-button" onClick={refreshAll} disabled={isRefreshing}>
              {isRefreshing ? "Refreshing..." : "Refresh live data"}
            </button>
            <button className="ghost-button" onClick={handleGenerateReport} disabled={isGeneratingReport}>
              {isGeneratingReport ? "Generating..." : "Generate report"}
            </button>
            <button className="primary-button" onClick={() => navigateTo("ask")}>
              Ask AskMamma
            </button>
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}
        {reportSummary ? <div className="info-banner">Report: {reportSummary}</div> : null}
        {forecastSummary ? <div className="info-banner">Forecast: {forecastSummary}</div> : null}

        {routeState.route === "dashboard" ? (
          <DashboardPage
            cards={dashboardCards}
            dashboard={dashboard}
            forecastAlerts={forecastAlerts}
            highDemandProducts={highDemandProducts}
            navigateTo={navigateTo}
          />
        ) : null}

        {routeState.route === "products" ? (
          <ProductsPage
            filter={activeProductFilter}
            onFilterChange={(filter) => navigateTo("products", filter === "all" ? {} : { filter })}
            productQuery={productQuery}
            setProductQuery={setProductQuery}
            products={visibleProducts}
            selectedProduct={selectedProduct}
            setSelectedProductId={setSelectedProductId}
          />
        ) : null}

        {routeState.route === "forecasts" ? (
          <ForecastsPage forecastAlerts={forecastAlerts} onOpenReorder={() => navigateTo("reorder")} />
        ) : null}

        {routeState.route === "reorder" ? <ReorderPage recommendations={recommendations} /> : null}
        {routeState.route === "suppliers" ? <SuppliersPage suppliers={suppliers} /> : null}

        {routeState.route === "reports" ? (
          <ReportsPage reports={reports} onGenerateReport={handleGenerateReport} isGeneratingReport={isGeneratingReport} />
        ) : null}

        {routeState.route === "ask" ? (
          <AskPage
            sessionId={sessionId}
            messages={messages}
            chatInput={chatInput}
            setChatInput={setChatInput}
            submitChatMessage={submitChatMessage}
            isSending={isSending}
          />
        ) : null}

        {routeState.route === "architecture" ? <ArchitecturePage onAsk={() => navigateTo("ask")} /> : null}
        {routeState.route === "admin" ? <AdminPage traces={traces} /> : null}
      </main>
    </div>
  );
}

function DashboardPage({ cards, dashboard, forecastAlerts, highDemandProducts, navigateTo }) {
  return (
    <div className="page-grid">
      <section className="hero-card">
        <div className="hero-copy">
          <span className="eyebrow">Business dashboard</span>
          <h3>Click a business category and jump straight to the right data.</h3>
          <p>
            AskMamma is organized around operational decisions, not raw tables. Choose inventory, forecasts,
            suppliers, reports, AI chat, or architecture and go directly to the filtered view you need.
          </p>
        </div>

        <div className="mini-signal-panel">
          <div className="panel-stat">
            <span>Forecast alerts</span>
            <strong>{formatter.format(forecastAlerts.length)}</strong>
          </div>
          <div className="panel-stat">
            <span>High demand products</span>
            <strong>{formatter.format(highDemandProducts.length)}</strong>
          </div>
          <div className="panel-stat">
            <span>Recent AI actions</span>
            <strong>{formatter.format((dashboard?.recent_ai_actions ?? []).length)}</strong>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Navigation shortcuts</p>
            <h3>Business categories</h3>
          </div>
        </div>

        <div className="dashboard-card-grid">
          {cards.map((card) => (
            <article key={card.title} className={`dashboard-card dashboard-card-${card.accent}`}>
              <div>
                <p className="section-label">{card.title}</p>
                <strong>{typeof card.count === "number" ? `${card.count} items` : card.count}</strong>
                <p>{card.description}</p>
              </div>
              <button className="inline-link" onClick={() => navigateTo(card.route, card.params ?? {})}>
                {card.action}
              </button>
            </article>
          ))}
        </div>
      </section>

      <section className="panel split-panel">
        <div>
          <p className="section-label">How AskMamma Works</p>
          <h3>Three steps</h3>
          <ol className="step-list">
            <li>Ask a business question.</li>
            <li>AskMamma routes it to the correct agent.</li>
            <li>The agent uses tools, data, and reports to return the answer.</li>
          </ol>
        </div>

        <div className="watch-card">
          <span>Current business signals</span>
          <div className="watch-row">
            <strong>Low stock pressure</strong>
            <p>{formatter.format(dashboard?.low_stock_products ?? 0)} items are at or below threshold.</p>
          </div>
          <div className="watch-row">
            <strong>Out-of-stock exposure</strong>
            <p>{formatter.format(dashboard?.out_of_stock_products ?? 0)} items are currently unavailable.</p>
          </div>
          <div className="watch-row">
            <strong>Demand watchlist</strong>
            <p>{formatter.format(highDemandProducts.length)} products show strong recent demand signals.</p>
          </div>
        </div>
      </section>
    </div>
  );
}

function ProductsPage({ filter, onFilterChange, productQuery, setProductQuery, products, selectedProduct, setSelectedProductId }) {
  const filterPills = [
    { key: "all", label: "All Products" },
    { key: "low-stock", label: "Low Stock" },
    { key: "out-of-stock", label: "Out of Stock" },
    { key: "high-demand", label: "High Demand" },
  ];

  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Products</p>
            <h3>{filterPills.find((item) => item.key === filter)?.label ?? "Products"}</h3>
          </div>
          <div className="toolbar-actions">
            <input
              className="search-input"
              value={productQuery}
              onChange={(event) => setProductQuery(event.target.value)}
              placeholder="Search SKU, product, category, supplier..."
            />
          </div>
        </div>

        <div className="pill-row">
          {filterPills.map((pill) => (
            <button
              key={pill.key}
              className={`filter-pill ${filter === pill.key ? "filter-pill-active" : ""}`}
              onClick={() => onFilterChange(pill.key)}
            >
              {pill.label}
            </button>
          ))}
        </div>

        <div className="table-layout">
          <div className="table-card">
            <div className="table-toolbar">
              <span>{products.length} products shown</span>
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
                      onClick={() => setSelectedProductId(product.id)}
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
              {!products.length ? <div className="empty-state">No products matched this filter.</div> : null}
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
              <div className="empty-state">Choose an item to review its details.</div>
            )}
          </aside>
        </div>
      </section>
    </div>
  );
}

function ForecastsPage({ forecastAlerts, onOpenReorder }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Forecasts</p>
            <h3>Forecast alerts</h3>
          </div>
          <button className="secondary-button" onClick={onOpenReorder}>
            Open reorder recommendations
          </button>
        </div>

        <div className="info-banner">
          Forecasts are calculated using historical demo sales/movement data and forecasting logic. AI may explain the
          result, but it does not invent stock or forecast numbers.
        </div>

        <div className="list-stack">
          {forecastAlerts.map((item) => (
            <article key={item.item_id} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{item.name}</strong>
                  <span>
                    {item.sku} · {item.supplier}
                  </span>
                </div>
                <strong>{formatter.format(item.recommended_quantity)} units</strong>
              </div>
              <p>{item.reason}</p>
              <div className="meta-row">
                <span>Current stock: {formatter.format(item.current_stock)}</span>
                <span>Threshold: {formatter.format(item.reorder_level)}</span>
              </div>
            </article>
          ))}
          {!forecastAlerts.length ? <div className="empty-state">No forecast alerts were generated.</div> : null}
        </div>
      </section>
    </div>
  );
}

function ReorderPage({ recommendations }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Reorder</p>
            <h3>Reorder recommendations</h3>
          </div>
        </div>

        <div className="list-stack">
          {recommendations.map((item) => (
            <article key={item.item_id} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{item.name}</strong>
                  <span>
                    {item.sku} · {item.supplier}
                  </span>
                </div>
                <strong>{formatter.format(item.recommended_quantity)} recommended</strong>
              </div>
              <p>{item.reason}</p>
              <div className="meta-row">
                <span>Current stock: {formatter.format(item.current_stock)}</span>
                <span>Reorder level: {formatter.format(item.reorder_level)}</span>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function SuppliersPage({ suppliers }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Suppliers</p>
            <h3>Supplier overview</h3>
          </div>
        </div>

        <div className="supplier-grid">
          {suppliers.map((supplier) => (
            <article key={supplier.id} className="supplier-card">
              <div className="list-card-top">
                <div>
                  <strong>{supplier.name}</strong>
                  <span>{supplier.country || "No country listed"}</span>
                </div>
                <strong>{supplier.product_count} products</strong>
              </div>
              <p>{supplier.contact_email || "No contact email listed."}</p>
              <div className="meta-row">
                <span>Low stock items: {supplier.low_stock_count}</span>
                <span>Out of stock: {supplier.out_of_stock_count}</span>
              </div>
              <div className="meta-row">
                <span>Lead time: {supplier.lead_time_days} days</span>
                <span>{supplier.phone || "No phone listed"}</span>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function ReportsPage({ reports, onGenerateReport, isGeneratingReport }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Reports</p>
            <h3>Generated reports</h3>
          </div>
          <button className="primary-button" onClick={onGenerateReport} disabled={isGeneratingReport}>
            {isGeneratingReport ? "Generating..." : "Generate new report"}
          </button>
        </div>

        <div className="list-stack">
          {reports.map((report) => (
            <article key={report.path} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{report.file_name}</strong>
                  <span>{new Date(report.updated_at).toLocaleString()}</span>
                </div>
                <strong>{formatter.format(report.size_bytes)} bytes</strong>
              </div>
              <p>{report.path}</p>
            </article>
          ))}
          {!reports.length ? <div className="empty-state">No reports have been generated yet.</div> : null}
        </div>
      </section>
    </div>
  );
}

function AskPage({ sessionId, messages, chatInput, setChatInput, submitChatMessage, isSending }) {
  return (
    <div className="page-grid">
      <section className="panel chat-panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Ask AskMamma</p>
            <h3>AI operations chat</h3>
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
                  {message.meta.tools_called?.length ? <span>Tools: {message.meta.tools_called.join(", ")}</span> : null}
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
            placeholder="Ask about availability, suppliers, forecasts, documents, or reports..."
            rows={4}
          />
          <button className="primary-button" type="submit" disabled={isSending || !chatInput.trim()}>
            {isSending ? "Sending..." : "Ask AskMamma"}
          </button>
        </form>
      </section>
    </div>
  );
}

function ArchitecturePage({ onAsk }) {
  const agentCards = [
    {
      title: "Supervisor Agent",
      description: "Decides what the user is asking and routes the request to the correct specialist.",
    },
    {
      title: "Inventory Agent",
      description: "Handles product lookup, stock levels, supplier details, availability, and reorder questions.",
    },
    {
      title: "Forecast Agent",
      description: "Uses historical sales or movement data and forecasting tools to calculate future demand and reorder recommendations.",
    },
    {
      title: "Document Agent",
      description: "Uses RAG and document search to answer questions from uploaded or internal documents.",
    },
    {
      title: "Report Agent",
      description: "Generates inventory, forecast, and operations reports.",
    },
    {
      title: "Quality Review Agent",
      description: "Checks the answer before it is returned to the user.",
    },
  ];

  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">AI Architecture</p>
            <h3>How the AskMamma agents work</h3>
          </div>
          <button className="secondary-button" onClick={onAsk}>
            Ask AskMamma now
          </button>
        </div>

        <div className="architecture-diagram">
          <div className="diagram-node">User request</div>
          <div className="diagram-arrow">↓</div>
          <div className="diagram-node diagram-highlight">Supervisor Agent</div>
          <div className="diagram-arrow">↓</div>
          <div className="diagram-grid">
            <div className="diagram-node">Inventory Agent</div>
            <div className="diagram-node">Forecast Agent</div>
            <div className="diagram-node">Document Agent</div>
            <div className="diagram-node">Report Agent</div>
          </div>
          <div className="diagram-arrow">↓</div>
          <div className="diagram-node">Quality Review Agent</div>
          <div className="diagram-arrow">↓</div>
          <div className="diagram-node diagram-final">Final Response</div>
        </div>

        <div className="supplier-grid architecture-grid">
          {agentCards.map((agent) => (
            <article key={agent.title} className="supplier-card">
              <strong>{agent.title}</strong>
              <p>{agent.description}</p>
            </article>
          ))}
        </div>

        <div className="watch-card architecture-note">
          <span>Forecasting flow</span>
          <p>Forecasting numbers must not be invented by the LLM.</p>
          <ol className="step-list">
            <li>Forecast Agent</li>
            <li>Inventory Lookup Tool</li>
            <li>Historical Sales / Movement Tool</li>
            <li>Forecast Calculation Tool</li>
            <li>Optional LLM Explanation</li>
          </ol>
          <p>
            The LLM may explain the forecast in plain English, but the actual numbers come from deterministic calculation
            and stored historical/demo data.
          </p>
        </div>

        <div className="watch-card architecture-note">
          <span>Example flow</span>
          <ol className="step-list">
            <li>Supervisor Agent detects inventory or reorder intent.</li>
            <li>Forecast Agent checks historical usage.</li>
            <li>Inventory Agent checks current stock.</li>
            <li>Forecast Calculation Tool calculates expected demand.</li>
            <li>Report or response is generated.</li>
            <li>Quality Review Agent checks the answer.</li>
          </ol>
        </div>
      </section>
    </div>
  );
}

function AdminPage({ traces }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Admin</p>
            <h3>Traces and technical activity</h3>
          </div>
        </div>

        <div className="list-stack">
          {traces.map((trace, index) => (
            <article key={`${trace.session_id}-${index}`} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{trace.selected_agent || "Agent"}</strong>
                  <span>{new Date(trace.created_at).toLocaleString()}</span>
                </div>
                <strong>{trace.latency_ms ? `${trace.latency_ms} ms` : "Trace"}</strong>
              </div>
              <p>{trace.user_input}</p>
              <div className="meta-row">
                <span>Session {trace.session_id.slice(0, 8)}</span>
                <span>{trace.final_answer ? "Answer captured" : "No final answer stored"}</span>
              </div>
            </article>
          ))}
          {!traces.length ? <div className="empty-state">No traces recorded yet.</div> : null}
        </div>
      </section>
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

function pageTitle(routeState) {
  const map = {
    dashboard: "Dashboard",
    products: routeState.params.filter === "high-demand" ? "High Demand Products" : "Products",
    forecasts: "Forecast Alerts",
    reorder: "Reorder Recommendations",
    suppliers: "Suppliers",
    reports: "Reports",
    ask: "Ask AskMamma",
    architecture: "AI Architecture",
    admin: "Admin / Traces",
  };
  return map[routeState.route] ?? "AskMamma";
}
