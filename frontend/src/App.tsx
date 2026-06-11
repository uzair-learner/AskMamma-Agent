import { startTransition, useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const SESSION_STORAGE_KEY = "Inventory Pilot AI-session-id";
const AUTH_STORAGE_KEY = "inventory-pilot-auth-token";
const USER_STORAGE_KEY = "inventory-pilot-auth-user";

type UserRole = "admin" | "manager" | "analyst" | "viewer";

interface AuthUser {
  id: number;
  username: string;
  full_name: string;
  role: UserRole;
  tenant_id: number;
  tenant_slug: string;
  tenant_name: string;
}

interface Product {
  id: number;
  sku: string;
  name: string;
  category: string;
  description?: string;
  supplier_id?: number | null;
  supplier_name?: string;
  stock_quantity: number;
  reorder_level: number;
  reorder_quantity: number;
  price: number;
  cost?: number;
  location?: string;
  expiry_date?: string | null;
  lead_time_days?: number;
  created_at?: string;
  updated_at?: string;
}

interface Supplier {
  id: number;
  name: string;
  country?: string;
  contact_email?: string;
  phone?: string;
  lead_time_days: number;
  product_count: number;
  low_stock_count: number;
  out_of_stock_count: number;
}

interface AIInsight {
  ai_explanation?: string;
  message?: string;
  ai_source?: string;
  provider?: string;
  model?: string;
  llm_used?: boolean;
  generated_at?: string;
}

interface ReorderRequest {
  id: number;
  supplier_id: number;
  supplier_name: string;
  status: "Draft" | "Submitted" | "Approved" | "Rejected" | "Completed";
  notes?: string;
  items: Array<Record<string, unknown>>;
  created_at: string;
  updated_at: string;
}

const formatter = new Intl.NumberFormat("en-US");
const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

const cannedPrompts = [
  "Which sample demo items are low in availability right now?",
  "Generate a short Inventory Pilot AI operations report.",
  "What does the return policy say about unopened items?",
];

const navigationItems = [
  { label: "Dashboard", route: "dashboard" },
  { label: "Inventory Management", route: "inventory" },
  { label: "Products", route: "products" },
  { label: "High Demand", route: "products", params: { filter: "high-demand" } },
  { label: "Forecasts", route: "forecasts" },
  { label: "Reorder", route: "reorder" },
  { label: "Suppliers", route: "suppliers" },
  { label: "Reports", route: "reports" },
  { label: "Inventory Pilot AI Assistant", route: "ask" },
  { label: "AI Architecture", route: "architecture" },
  { label: "Admin", route: "admin" },
];

const rolePermissions = {
  admin: new Set(["admin:read", "inventory:write", "report:write", "reorder:read"]),
  manager: new Set(["inventory:write", "report:write", "reorder:read"]),
  analyst: new Set(["reorder:read"]),
  viewer: new Set([]),
};

function can(user, permission) {
  return rolePermissions[user?.role]?.has(permission) ?? false;
}

const dashboardCardConfig = [
  ["Inventory Management", "inventory", {}, "mint", "Add, edit, delete, search, and inspect inventory records stored in SQLite.", "Manage inventory"],
  ["All Products", "products", {}, "blue", "Browse the full sample catalog, search items, and inspect product details.", "View all products"],
  ["Low Stock", "products", { filter: "low-stock" }, "amber", "Items below their current threshold and likely to need attention soon.", "View low-stock items"],
  ["Out of Stock", "products", { filter: "out-of-stock" }, "coral", "Products currently unavailable in the sample catalog.", "View unavailable items"],
  ["High Demand Products", "products", { filter: "high-demand" }, "mint", "Products with strong recent sales or usage signals.", "View high-demand items"],
  ["Forecast Alerts", "forecasts", {}, "violet", "Products where historical usage suggests near-term demand pressure.", "View forecast alerts"],
  ["Reorder Recommendations", "reorder", {}, "sunset", "Suggested purchasing actions based on current stock and deterministic forecasts.", "View reorder plan"],
  ["Suppliers", "suppliers", {}, "teal", "See supplier coverage, low-stock exposure, and demo partner details.", "View suppliers"],
  ["Reports", "reports", {}, "slate", "Generate and download Inventory Pilot AI operations reports.", "View reports"],
  ["Inventory Pilot AI Assistant", "ask", {}, "indigo", "Open the AI operations assistant for inventory, forecast, or document questions.", "Open AI chat"],
  ["AI Architecture", "architecture", {}, "rose", "Understand how the supervisor and specialist agents handle business requests.", "Explore agent flow"],
  ["Admin / Traces", "admin", {}, "graphite", "Review diagnostics, request metadata, and internal operational detail.", "Open admin traces"],
].map(([title, route, params, accent, description, action]) => ({ title, route, params, accent, description, action }));

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
  return { route: routePart || "dashboard", params: Object.fromEntries(new URLSearchParams(queryString).entries()) };
}

function navigateTo(route, params = {}) {
  const query = new URLSearchParams(params).toString();
  window.location.hash = query ? `${route}?${query}` : route;
}

function formatGeneratedAt(value) {
  if (!value) {
    return "Unavailable";
  }
  return new Date(value).toLocaleString();
}

function isVerifiedOllamaResponse(meta) {
  if (!meta) {
    return false;
  }
  const source = String(meta.ai_source || meta.provider || "").trim().toLowerCase();
  return source === "ollama" && Boolean(meta.llm_used) && Boolean(String(meta.model || "").trim());
}

async function request(path, options = {}) {
  const token = window.sessionStorage.getItem(AUTH_STORAGE_KEY);
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      const detail = payload.detail ?? payload.message;
      message = typeof detail === "string" ? detail : detail?.message ?? detail?.error_code ?? message;
    } catch {
      // Keep the default request error when the payload is not JSON.
    }
    throw new Error(message);
  }

  return response.json();
}

function productStatus(product: Product) {
  if (Number(product.stock_quantity) <= 0) {
    return "Out of stock";
  }
  if (Number(product.stock_quantity) <= Number(product.reorder_level)) {
    return "Low stock";
  }
  return "In stock";
}

function emptyInventoryForm() {
  return {
    id: null,
    name: "",
    sku: "",
    category: "",
    supplier_id: "",
    description: "",
    price: "",
    stock_quantity: "",
    reorder_level: "",
    reorder_quantity: "",
    location: "",
    expiry_date: "",
  };
}

function productToInventoryForm(product: Product) {
  return {
    id: product.id,
    name: product.name ?? "",
    sku: product.sku ?? "",
    category: product.category ?? "",
    supplier_id: product.supplier_id ? String(product.supplier_id) : "",
    description: product.description ?? "",
    price: String(product.price ?? ""),
    stock_quantity: String(product.stock_quantity ?? ""),
    reorder_level: String(product.reorder_level ?? ""),
    reorder_quantity: String(product.reorder_quantity ?? product.reorder_level ?? ""),
    location: product.location ?? "",
    expiry_date: product.expiry_date ?? "",
  };
}

async function loadDataPart(label, path, fallback) {
  try {
    return { label, ok: true, data: await request(path) };
  } catch (error) {
    return { label, ok: false, data: fallback, error };
  }
}

export default function App() {
  const [sessionId] = useState(getSessionId);
  const [authToken, setAuthToken] = useState(() => window.sessionStorage.getItem(AUTH_STORAGE_KEY) ?? "");
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(() => {
    const stored = window.sessionStorage.getItem(USER_STORAGE_KEY);
    return stored ? JSON.parse(stored) : null;
  });
  const [loginForm, setLoginForm] = useState({ username: "admin@example.com", password: "AdminPass123!" });
  const [loginError, setLoginError] = useState("");
  const [routeState, setRouteState] = useState(parseHash);
  const [health, setHealth] = useState({ status: "loading", environment: "..." });
  const [dashboard, setDashboard] = useState<any>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [reports, setReports] = useState<any[]>([]);
  const [reorderRequests, setReorderRequests] = useState<ReorderRequest[]>([]);
  const [diagnostics, setDiagnostics] = useState<any>(null);
  const [pageInsight, setPageInsight] = useState<AIInsight | null>(null);
  const [pageInsightLoading, setPageInsightLoading] = useState(false);
  const [selectedProductId, setSelectedProductId] = useState(null);
  const [productQuery, setProductQuery] = useState("");
  const [inventoryMessage, setInventoryMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
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
        "Welcome to Inventory Pilot AI. Ask about inventory, forecasting, reorder recommendations, suppliers, reports, or documents.",
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

  async function loadAllData(user = currentUser) {
    const results = await Promise.all([
      loadDataPart("health", "/health", { status: "unavailable", environment: "..." }),
      loadDataPart("dashboard", "/dashboard", null),
      loadDataPart("products", "/demo/items?limit=100", []),
      loadDataPart("suppliers", "/demo/suppliers", []),
      loadDataPart("reorder recommendations", "/demo/recommendations/reorder", []),
      loadDataPart("reports", "/reports", []),
      can(user, "reorder:read")
        ? loadDataPart("reorder requests", "/reorder/requests", [])
        : Promise.resolve({ label: "reorder requests", ok: true, data: [] }),
      can(user, "admin:read")
        ? loadDataPart("admin diagnostics", "/admin/diagnostics", null)
        : Promise.resolve({ label: "admin diagnostics", ok: true, data: null }),
    ]);
    const [healthData, dashboardData, productData, supplierData, recommendationData, reportData, reorderRequestData, diagnosticsData] =
      results.map((result) => result.data);

    startTransition(() => {
      setHealth(healthData);
      setDashboard(dashboardData);
      setProducts(productData);
      setSuppliers(supplierData);
      setRecommendations(recommendationData);
      setReports(reportData);
      setReorderRequests(reorderRequestData);
      setDiagnostics(diagnosticsData);
      setSelectedProductId((current) => current ?? productData[0]?.id ?? null);
    });

    const failed = results.filter((result) => !result.ok);
    if (failed.length) {
      throw new Error(
        failed
          .map((result) => `${result.label}: ${result.error instanceof Error ? result.error.message : "Request failed"}`)
          .join("; "),
      );
    }
  }

  useEffect(() => {
    if (authToken && currentUser) {
      loadAllData().catch((loadError) => setError(loadError.message));
    }
  }, [authToken, currentUser?.id, currentUser?.role]);

  async function handleLogin(event) {
    event.preventDefault();
    setLoginError("");
    try {
      const result = await request("/auth/login", {
        method: "POST",
        body: JSON.stringify(loginForm),
      });
      window.sessionStorage.setItem(AUTH_STORAGE_KEY, result.access_token);
      window.sessionStorage.setItem(USER_STORAGE_KEY, JSON.stringify(result.user));
      setCurrentUser(result.user);
      setAuthToken(result.access_token);
    } catch (loginFailure) {
      setLoginError(loginFailure.message);
    }
  }

  async function handleLogout() {
    try {
      await request("/auth/logout", { method: "POST" });
    } catch {
      // Local logout still clears demo session state if the server session is already gone.
    }
    window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
    window.sessionStorage.removeItem(USER_STORAGE_KEY);
    setAuthToken("");
    setCurrentUser(null);
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content:
          "Welcome to Inventory Pilot AI. Ask about inventory, forecasting, reorder recommendations, suppliers, reports, or documents.",
      },
    ]);
  }

  useEffect(() => {
    let cancelled = false;
    const path = insightPath(routeState, selectedProductId);
    if (!path) {
      setPageInsight(null);
      setPageInsightLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setPageInsightLoading(true);
    request(path)
      .then((insight) => {
        if (!cancelled) {
          startTransition(() => {
            setPageInsight(insight);
            setPageInsightLoading(false);
          });
        }
      })
      .catch((loadError) => {
        if (!cancelled) {
          startTransition(() => {
            setPageInsight({
              ai_explanation: "Ollama is unavailable. AI content was not generated.",
              ai_source: "Ollama",
              provider: "Ollama",
              model: diagnostics?.configured_model ?? diagnostics?.model ?? "Unavailable",
              llm_used: false,
              generated_at: new Date().toISOString(),
            });
            setPageInsightLoading(false);
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    routeState.route,
    routeState.params.filter,
    selectedProductId,
    reports.length,
    recommendations.length,
    suppliers.length,
    products.length,
    dashboard?.predicted_high_demand_products?.length,
  ]);

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
      ask: Math.max(0, messages.length - 1),
      architecture: 5,
      admin: diagnostics?.recent_requests?.length ?? 0,
    }),
    [diagnostics?.recent_requests?.length, forecastAlerts.length, highDemandProducts.length, messages.length, products, recommendations.length, reports.length, suppliers.length],
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
  const selectedInventoryProduct = products.find((item) => item.id === selectedProductId) ?? products[0] ?? null;

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
      const report = await request("/reports/inventory-pilot-ai");
      setReportSummary("");
      startTransition(() => setReports((current) => [report, ...current.filter((item) => item.created_at !== report.created_at)]));
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

  function validateInventoryForm(form) {
    const errors: Record<string, string> = {};
    if (!form.name.trim()) {
      errors.name = "Name is required.";
    }
    if (!form.sku.trim()) {
      errors.sku = "SKU is required.";
    }
    const numericFields = [
      ["stock_quantity", "Quantity on hand"],
      ["price", "Unit price"],
      ["reorder_level", "Reorder level"],
      ["reorder_quantity", "Reorder quantity"],
    ];
    numericFields.forEach(([field, label]) => {
      const value = Number(form[field]);
      if (form[field] === "" || Number.isNaN(value)) {
        errors[field] = `${label} must be numeric.`;
      } else if (value < 0) {
        errors[field] = `${label} cannot be negative.`;
      }
    });
    return errors;
  }

  function inventoryPayloadFromForm(form) {
    const price = Number(form.price);
    return {
      sku: form.sku.trim(),
      name: form.name.trim(),
      category: form.category.trim() || "Uncategorized",
      description: form.description.trim(),
      supplier_id: form.supplier_id ? Number(form.supplier_id) : null,
      price,
      cost: price,
      stock_quantity: Number(form.stock_quantity),
      reorder_level: Number(form.reorder_level),
      reorder_quantity: Number(form.reorder_quantity || form.reorder_level || 0),
      location: form.location.trim() || null,
      expiry_date: form.expiry_date || null,
      confirm: true,
    };
  }

  async function handleSaveInventoryProduct(form) {
    setError("");
    setInventoryMessage(null);
    try {
      const payload = inventoryPayloadFromForm(form);
      const saved = await request(form.id ? `/demo/items/${form.id}` : "/demo/items", {
        method: form.id ? "PUT" : "POST",
        body: JSON.stringify(payload),
      });
      setSelectedProductId(saved.id);
      await loadAllData();
      setInventoryMessage({ type: "success", text: `${saved.name} was ${form.id ? "updated" : "added"} successfully.` });
      return saved;
    } catch (saveError) {
      console.error("Inventory save failed", saveError);
      const message = saveError instanceof Error ? saveError.message : "Inventory record could not be saved.";
      setInventoryMessage({ type: "error", text: message });
      throw saveError;
    }
  }

  async function handleDeleteInventoryProduct(product) {
    if (!product) {
      return;
    }
    const confirmed = window.confirm(`Delete ${product.name}? This will remove the inventory record from SQLite.`);
    if (!confirmed) {
      return;
    }
    setError("");
    setInventoryMessage(null);
    try {
      await request(`/demo/items/${product.id}?confirm=true`, { method: "DELETE" });
      await loadAllData();
      setSelectedProductId((current) => (current === product.id ? null : current));
      setInventoryMessage({ type: "success", text: `${product.name} was deleted successfully.` });
    } catch (deleteError) {
      console.error("Inventory delete failed", deleteError);
      const message = deleteError instanceof Error ? deleteError.message : "Inventory record could not be deleted.";
      setInventoryMessage({ type: "error", text: message });
    }
  }

  async function submitChatMessage(message) {
    if (!message.trim()) {
      return;
    }
    setMessages((current) => [...current, { id: `user-${Date.now()}`, role: "user", content: message }]);
    setChatInput("");
    setIsSending(true);
    setError("");

    try {
      const result = await request("/agent/chat", {
        method: "POST",
        body: JSON.stringify({ message, session_id: sessionId }),
      });
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: result.answer,
          meta: {
            ai_source: result.ai_source,
            provider: result.provider,
            model: result.model,
            llm_used: result.llm_used,
            generated_at: result.generated_at,
            selected_agent: result.selected_agent,
            tools_called: result.tools_called ?? [],
            response_time_ms: result.response_time_ms ?? result.latency_ms ?? 0,
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
            ai_source: "Ollama",
            provider: "",
            model: "",
            llm_used: false,
            generated_at: new Date().toISOString(),
            selected_agent: "SupervisorAgent",
            tools_called: [],
            response_time_ms: 0,
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
        "Inventory Pilot AI Assistant": productCounts.ask,
        "AI Architecture": productCounts.architecture,
        "Admin / Traces": productCounts.admin,
        "Inventory Management": productCounts.all,
      }[card.title] ?? "--",
  }));

  if (!authToken || !currentUser) {
    return (
      <LoginScreen
        loginForm={loginForm}
        setLoginForm={setLoginForm}
        loginError={loginError}
        onLogin={handleLogin}
      />
    );
  }

  return (
    <div className="workspace-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <aside className="sidebar">
        <div className="brand-block">
          <span className="eyebrow">Inventory Pilot AI</span>
          <h1>Inventory Pilot AI</h1>
          <p>Move by business category instead of hunting through a big table first.</p>
        </div>
        <nav className="sidebar-nav">
          {navigationItems
            .filter((item) => item.route !== "admin" || can(currentUser, "admin:read"))
            .map((item) => {
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
            <p className="section-label">Inventory Pilot AI</p>
            <h2>{pageTitle(routeState)}</h2>
          </div>
          <div className="topbar-actions">
            <span className="session-pill">
              {currentUser.role} | {currentUser.tenant_slug}
            </span>
            <button className="ghost-button" onClick={handleLogout}>
              Logout
            </button>
            <button className="secondary-button" onClick={refreshAll} disabled={isRefreshing}>
              {isRefreshing ? "Refreshing..." : "Refresh live data"}
            </button>
            {can(currentUser, "report:write") ? (
              <button className="ghost-button" onClick={handleGenerateReport} disabled={isGeneratingReport}>
                {isGeneratingReport ? "Generating..." : "Generate report"}
              </button>
            ) : null}
            <button className="primary-button" onClick={() => navigateTo("ask")}>
              Inventory Pilot AI Assistant
            </button>
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}
        {reportSummary ? <div className="info-banner">{reportSummary}</div> : null}
        {forecastSummary ? <div className="info-banner">{forecastSummary}</div> : null}

        {routeState.route === "dashboard" ? (
          <DashboardPage
            cards={dashboardCards}
            dashboard={dashboard}
            forecastAlerts={forecastAlerts}
            highDemandProducts={highDemandProducts}
            insight={pageInsight}
            loading={pageInsightLoading}
          />
        ) : null}
        {routeState.route === "inventory" ? (
          <InventoryManagementPage
            products={products}
            suppliers={suppliers}
            selectedProduct={selectedInventoryProduct}
            setSelectedProductId={setSelectedProductId}
            onSave={handleSaveInventoryProduct}
            onDelete={handleDeleteInventoryProduct}
            onRefresh={refreshAll}
            isRefreshing={isRefreshing}
            canWrite={can(currentUser, "inventory:write")}
            message={inventoryMessage}
            clearMessage={() => setInventoryMessage(null)}
            validateForm={validateInventoryForm}
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
            insight={pageInsight}
            loading={pageInsightLoading}
          />
        ) : null}
        {routeState.route === "forecasts" ? (
          <ForecastsPage
            forecastAlerts={forecastAlerts}
            onOpenReorder={() => navigateTo("reorder")}
            insight={pageInsight}
            loading={pageInsightLoading}
          />
        ) : null}
        {routeState.route === "reorder" ? <ReorderPage recommendations={recommendations} insight={pageInsight} loading={pageInsightLoading} /> : null}
        {routeState.route === "suppliers" ? <SuppliersPage suppliers={suppliers} reorderRequests={reorderRequests} insight={pageInsight} loading={pageInsightLoading} /> : null}
        {routeState.route === "reports" ? (
          <ReportsPage
            reports={reports}
            onGenerateReport={handleGenerateReport}
            isGeneratingReport={isGeneratingReport}
            canGenerateReport={can(currentUser, "report:write")}
            insight={pageInsight}
            loading={pageInsightLoading}
          />
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
        {routeState.route === "admin" ? <AdminPage diagnostics={diagnostics} /> : null}
      </main>
    </div>
  );
}

function LoginScreen({ loginForm, setLoginForm, loginError, onLogin }) {
  return (
    <main className="login-shell">
      <section className="panel login-panel">
        <div>
          <p className="section-label">Inventory Pilot AI</p>
          <h1>Sign in</h1>
        </div>
        <form className="composer" onSubmit={onLogin}>
          <input
            className="search-input"
            value={loginForm.username}
            onChange={(event) => setLoginForm((current) => ({ ...current, username: event.target.value }))}
            placeholder="Username"
            autoComplete="username"
          />
          <input
            className="search-input"
            type="password"
            value={loginForm.password}
            onChange={(event) => setLoginForm((current) => ({ ...current, password: event.target.value }))}
            placeholder="Password"
            autoComplete="current-password"
          />
          {loginError ? <div className="error-banner">{loginError}</div> : null}
          <button className="primary-button" type="submit">
            Login
          </button>
        </form>
        <div className="info-banner">
          Demo users: admin@example.com, manager@example.com, analyst@example.com, viewer@example.com
        </div>
      </section>
    </main>
  );
}

function DataSourceBadge({ text, tone = "local" }) {
  return <div className={`data-source-badge data-source-${tone}`}>{text}</div>;
}

function AIProofPanel({ meta }) {
  const verified = isVerifiedOllamaResponse(meta);
  return (
    <div className="meta-row">
      <span>AI Source: {verified ? "Ollama" : "Not generated by Ollama"}</span>
      <span>Model: {verified ? meta.model : (meta.model || "Unavailable")}</span>
      <span>LLM Used: {verified ? "Yes" : "No"}</span>
      <span>Generated At: {verified ? formatGeneratedAt(meta.generated_at) : "Unavailable"}</span>
    </div>
  );
}

function AIInsightPanel({ insight, loading = false }) {
  const text = insight?.ai_explanation ?? insight?.message;
  if (loading) {
    return (
      <div className="watch-card architecture-note">
        <DataSourceBadge text="Checking Ollama status..." tone="local" />
        <p>Inventory Pilot AI is checking whether Ollama generated this AI section.</p>
        <AIProofPanel
          meta={{
            ai_source: "Ollama",
            provider: "Ollama",
            model: "",
            llm_used: false,
            generated_at: "",
          }}
        />
      </div>
    );
  }
  if (!text) {
    return (
      <div className="watch-card architecture-note">
        <DataSourceBadge text="No AI response generated yet" tone="local" />
        <p>No Ollama response has been generated for this section yet.</p>
        <AIProofPanel
          meta={{
            ai_source: "Ollama",
            provider: "Ollama",
            model: "",
            llm_used: false,
            generated_at: "",
          }}
        />
      </div>
    );
  }
  const badgeText = insight.llm_used
    ? `AI explanation via ${insight.provider} ${insight.model}`
    : "AI explanation unavailable";
  return (
    <div className="watch-card architecture-note">
      <DataSourceBadge text={badgeText} tone={insight.llm_used ? "ai" : "local"} />
      <p>{text}</p>
      <AIProofPanel meta={insight} />
    </div>
  );
}

function DashboardPage({ cards, dashboard, forecastAlerts, highDemandProducts, insight, loading }) {
  return (
    <div className="page-grid">
      <section className="hero-card">
        <div className="hero-copy">
          <span className="eyebrow">Business dashboard</span>
          <h3>Click a business category and jump straight to the right data.</h3>
          <p>
            Inventory Pilot AI is organized around operational decisions, not raw tables. Choose inventory, forecasts, suppliers,
            reports, AI chat, or architecture and go directly to the filtered view you need.
          </p>
          <DataSourceBadge text="Calculated from local inventory/demo data" />
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
      <AIInsightPanel insight={insight} loading={loading} />

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
          <p className="section-label">How Inventory Pilot AI Works</p>
          <h3>Three steps</h3>
          <ol className="step-list">
            <li>Ask a business question.</li>
            <li>Inventory Pilot AI routes it to the correct agent.</li>
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

function InventoryManagementPage({
  products,
  suppliers,
  selectedProduct,
  setSelectedProductId,
  onSave,
  onDelete,
  onRefresh,
  isRefreshing,
  canWrite,
  message,
  clearMessage,
  validateForm,
}) {
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [form, setForm] = useState(emptyInventoryForm);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [isSaving, setIsSaving] = useState(false);

  const filteredProducts = useMemo(() => {
    const search = query.trim().toLowerCase();
    return products.filter((product) => {
      const status = productStatus(product);
      const statusMatches =
        statusFilter === "all" ||
        (statusFilter === "in-stock" && status === "In stock") ||
        (statusFilter === "low-stock" && status === "Low stock") ||
        (statusFilter === "out-of-stock" && status === "Out of stock");
      if (!statusMatches) {
        return false;
      }
      if (!search) {
        return true;
      }
      return [product.name, product.sku, product.category, product.supplier_name, status]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(search));
    });
  }, [products, query, statusFilter]);

  const editingProduct = form.id ? products.find((product) => product.id === form.id) : null;

  function updateForm(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
    setFormErrors((current) => {
      const next = { ...current };
      delete next[field];
      return next;
    });
  }

  function startAdd() {
    setForm(emptyInventoryForm());
    setFormErrors({});
    clearMessage();
  }

  function startEdit(product) {
    setSelectedProductId(product.id);
    setForm(productToInventoryForm(product));
    setFormErrors({});
    clearMessage();
  }

  async function submitForm(event) {
    event.preventDefault();
    const nextErrors = validateForm(form);
    setFormErrors(nextErrors);
    if (Object.keys(nextErrors).length) {
      return;
    }
    setIsSaving(true);
    try {
      const saved = await onSave(form);
      setForm(productToInventoryForm(saved));
    } catch {
      // The parent records a user-friendly page message and logs the technical error.
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Inventory Management</p>
            <h3>Add, edit, delete, and search records</h3>
          </div>
          <div className="toolbar-actions">
            <button className="secondary-button" onClick={onRefresh} disabled={isRefreshing}>
              {isRefreshing ? "Refreshing..." : "Refresh"}
            </button>
            <button className="primary-button" onClick={startAdd} disabled={!canWrite}>
              Add product
            </button>
          </div>
        </div>
        <DataSourceBadge text="Inventory records are stored in the local SQLite database" />
        {message ? (
          <div className={message.type === "success" ? "info-banner message-banner" : "error-banner message-banner"}>
            <span>{message.text}</span>
            <button className="inline-link" onClick={clearMessage}>Clear</button>
          </div>
        ) : null}
        {!canWrite ? <div className="info-banner">Your role can view inventory records, but cannot change them.</div> : null}
        <div className="inventory-layout">
          <div className="inventory-list-area">
            <div className="table-toolbar inventory-toolbar">
              <input
                className="search-input"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search name, SKU, category, supplier, or stock status..."
              />
              <select className="form-input compact-select" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                <option value="all">All stock statuses</option>
                <option value="in-stock">In stock</option>
                <option value="low-stock">Low stock</option>
                <option value="out-of-stock">Out of stock</option>
              </select>
            </div>
            <div className="table-card">
              <div className="table-toolbar">
                <span>{filteredProducts.length} records shown</span>
                <span>{products.length} total records</span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Product</th>
                      <th>SKU</th>
                      <th>Category</th>
                      <th>Supplier</th>
                      <th>Status</th>
                      <th>Qty</th>
                      <th>Price</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredProducts.map((product) => (
                      <tr key={product.id} className={selectedProduct?.id === product.id ? "selected-row" : ""} onClick={() => setSelectedProductId(product.id)}>
                        <td><strong>{product.name}</strong></td>
                        <td>{product.sku}</td>
                        <td>{product.category}</td>
                        <td>{product.supplier_name ?? "Unassigned"}</td>
                        <td><span className={`stock-badge stock-${productStatus(product).toLowerCase().replace(/\s+/g, "-")}`}>{productStatus(product)}</span></td>
                        <td>{formatter.format(product.stock_quantity)}</td>
                        <td>{currencyFormatter.format(product.price)}</td>
                        <td>
                          <div className="row-actions">
                            <button className="inline-link" onClick={(event) => { event.stopPropagation(); startEdit(product); }} disabled={!canWrite}>
                              Edit
                            </button>
                            <button className="inline-link danger-link" onClick={(event) => { event.stopPropagation(); onDelete(product); }} disabled={!canWrite}>
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!filteredProducts.length ? <div className="empty-state">No inventory records match the current search or status filter.</div> : null}
              </div>
            </div>
          </div>

          <aside className="detail-card">
            <div className="section-label">Details</div>
            {selectedProduct ? (
              <>
                <h3>{selectedProduct.name}</h3>
                <p>{selectedProduct.description || "No description provided."}</p>
                <dl className="detail-grid">
                  <Detail label="Internal ID" value={selectedProduct.id} />
                  <Detail label="SKU" value={selectedProduct.sku} />
                  <Detail label="Category" value={selectedProduct.category} />
                  <Detail label="Supplier" value={selectedProduct.supplier_name ?? "Unassigned"} />
                  <Detail label="Lead Time" value={selectedProduct.lead_time_days ? `${selectedProduct.lead_time_days} days` : "N/A"} />
                  <Detail label="Status" value={productStatus(selectedProduct)} />
                  <Detail label="Quantity" value={formatter.format(selectedProduct.stock_quantity)} />
                  <Detail label="Reorder Level" value={formatter.format(selectedProduct.reorder_level)} />
                  <Detail label="Unit Price" value={currencyFormatter.format(selectedProduct.price)} />
                  <Detail label="Created" value={formatGeneratedAt(selectedProduct.created_at)} />
                  <Detail label="Updated" value={formatGeneratedAt(selectedProduct.updated_at)} />
                </dl>
                {canWrite ? (
                  <div className="toolbar-actions">
                    <button className="secondary-button" onClick={() => startEdit(selectedProduct)}>Edit selected</button>
                    <button className="ghost-button danger-button" onClick={() => onDelete(selectedProduct)}>Delete selected</button>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="empty-state">Select a record to view details.</div>
            )}
          </aside>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">{form.id ? "Edit Product" : "Add Product"}</p>
            <h3>{editingProduct ? editingProduct.name : "Inventory record"}</h3>
          </div>
        </div>
        <form className="inventory-form" onSubmit={submitForm}>
          <Field label="Name" error={formErrors.name}>
            <input className="form-input" value={form.name} onChange={(event) => updateForm("name", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="SKU" error={formErrors.sku}>
            <input className="form-input" value={form.sku} onChange={(event) => updateForm("sku", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Category">
            <input className="form-input" value={form.category} onChange={(event) => updateForm("category", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Supplier">
            <select className="form-input" value={form.supplier_id} onChange={(event) => updateForm("supplier_id", event.target.value)} disabled={!canWrite}>
              <option value="">Unassigned</option>
              {suppliers.map((supplier) => (
                <option key={supplier.id} value={supplier.id}>{supplier.name}</option>
              ))}
            </select>
          </Field>
          <Field label="Unit Price" error={formErrors.price}>
            <input className="form-input" type="number" min="0" step="0.01" value={form.price} onChange={(event) => updateForm("price", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Quantity On Hand" error={formErrors.stock_quantity}>
            <input className="form-input" type="number" min="0" step="1" value={form.stock_quantity} onChange={(event) => updateForm("stock_quantity", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Reorder Level" error={formErrors.reorder_level}>
            <input className="form-input" type="number" min="0" step="1" value={form.reorder_level} onChange={(event) => updateForm("reorder_level", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Reorder Quantity" error={formErrors.reorder_quantity}>
            <input className="form-input" type="number" min="0" step="1" value={form.reorder_quantity} onChange={(event) => updateForm("reorder_quantity", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Location">
            <input className="form-input" value={form.location} onChange={(event) => updateForm("location", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Expiry Date">
            <input className="form-input" type="date" value={form.expiry_date} onChange={(event) => updateForm("expiry_date", event.target.value)} disabled={!canWrite} />
          </Field>
          <Field label="Description">
            <textarea className="form-input form-textarea" value={form.description} onChange={(event) => updateForm("description", event.target.value)} disabled={!canWrite} />
          </Field>
          <div className="form-actions">
            <button className="secondary-button" type="button" onClick={startAdd}>
              Clear form
            </button>
            <button className="primary-button" type="submit" disabled={!canWrite || isSaving}>
              {isSaving ? "Saving..." : form.id ? "Update product" : "Add product"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

function Field({ label, error, children }) {
  return (
    <label className="form-field">
      <span>{label}</span>
      {children}
      {error ? <strong className="field-error">{error}</strong> : null}
    </label>
  );
}

function ProductsPage({ filter, onFilterChange, productQuery, setProductQuery, products, selectedProduct, setSelectedProductId, insight, loading }) {
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
            <input className="search-input" value={productQuery} onChange={(event) => setProductQuery(event.target.value)} placeholder="Search SKU, product, category, supplier..." />
          </div>
        </div>
        <DataSourceBadge text="Calculated from local inventory/demo data" />
        <AIInsightPanel insight={insight} loading={loading} />
        <div className="pill-row">
          {filterPills.map((pill) => (
            <button key={pill.key} className={`filter-pill ${filter === pill.key ? "filter-pill-active" : ""}`} onClick={() => onFilterChange(pill.key)}>
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
                    <tr key={product.id} className={selectedProduct?.id === product.id ? "selected-row" : ""} onClick={() => setSelectedProductId(product.id)}>
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

function ForecastsPage({ forecastAlerts, onOpenReorder, insight, loading }) {
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
        <DataSourceBadge text="Calculated from local inventory/demo data" />
        <div className="section-label">AI Forecast Explanation</div>
        <AIInsightPanel insight={insight} loading={loading} />
        <div className="info-banner">
          Forecasts are calculated using historical demo sales or movement data and forecasting logic. AI may explain the result, but it does not invent stock or forecast numbers.
        </div>
        <div className="list-stack">
          {forecastAlerts.map((item) => (
            <article key={item.item_id} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{item.name}</strong>
                  <span>{item.sku} | {item.supplier}</span>
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

function ReorderPage({ recommendations, insight, loading }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Reorder</p>
            <h3>Reorder recommendations</h3>
          </div>
        </div>
        <DataSourceBadge text="Calculated from local inventory/demo data" />
        <div className="section-label">AI Reorder Explanation</div>
        <AIInsightPanel insight={insight} loading={loading} />
        <div className="list-stack">
          {recommendations.map((item) => (
            <article key={item.item_id} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{item.name}</strong>
                  <span>{item.sku} | {item.supplier}</span>
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

function SuppliersPage({ suppliers, reorderRequests, insight, loading }: { suppliers: Supplier[]; reorderRequests: ReorderRequest[]; insight: AIInsight | null; loading: boolean }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Suppliers</p>
            <h3>Supplier overview</h3>
          </div>
        </div>
        <DataSourceBadge text="Calculated from local inventory/demo data" />
        <AIInsightPanel insight={insight} loading={loading} />
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
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Supplier Coordination</p>
            <h3>Reorder requests</h3>
          </div>
        </div>
        <div className="list-stack">
          {reorderRequests.map((request) => (
            <article key={request.id} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{request.supplier_name}</strong>
                  <span>Request #{request.id} | {formatGeneratedAt(request.created_at)}</span>
                </div>
                <strong>{request.status}</strong>
              </div>
              <p>{request.notes || "No notes provided."}</p>
              <div className="meta-row">
                <span>Items: {request.items.length}</span>
                <span>Updated: {formatGeneratedAt(request.updated_at)}</span>
              </div>
            </article>
          ))}
          {!reorderRequests.length ? <div className="empty-state">No supplier reorder requests yet.</div> : null}
        </div>
      </section>
    </div>
  );
}

function ReportsPage({ reports, onGenerateReport, isGeneratingReport, canGenerateReport, insight, loading }) {
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Reports</p>
            <h3>Online reports</h3>
          </div>
          {canGenerateReport ? (
            <button className="primary-button" onClick={onGenerateReport} disabled={isGeneratingReport}>
              {isGeneratingReport ? "Generating..." : "Generate new report"}
            </button>
          ) : null}
        </div>
        <DataSourceBadge text="Generated online with Ollama" tone="ai" />
        <AIInsightPanel insight={insight} loading={loading} />
        <div className="list-stack">
          {reports.map((report, index) => (
            <article key={`${report.created_at || report.generated_at}-${index}`} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{report.title || "Online report"}</strong>
                  <span>{formatGeneratedAt(report.created_at || report.generated_at)}</span>
                </div>
                <strong>{report.status || (report.llm_used ? "success" : "failed")}</strong>
              </div>
              <p>{report.report_content || report.response || "Ollama is unavailable. Report was not generated."}</p>
              <AIProofPanel meta={{ ...report, generated_at: report.created_at || report.generated_at }} />
              {report.error_message ? <p>{report.error_message}</p> : null}
            </article>
          ))}
          {!reports.length ? <div className="empty-state">No online reports have been generated yet.</div> : null}
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
            <p className="section-label">Inventory Pilot AI Assistant</p>
            <h3>Inventory Pilot AI Assistant</h3>
          </div>
          <span className="session-pill">Session {sessionId.slice(0, 8)}</span>
        </div>
        <DataSourceBadge text="Generated using AI explanation" tone="ai" />
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
              <div className="message-role">{message.role === "assistant" ? "Inventory Pilot AI Assistant" : "You"}</div>
              <p>{message.content}</p>
              {message.role === "assistant" && message.meta ? <AIProofPanel meta={message.meta} /> : null}
              {message.meta ? (
                <details className="technical-details">
                  <summary>Technical Details</summary>
                  <div className="technical-grid">
                    <span>AI Source</span><strong>{isVerifiedOllamaResponse(message.meta) ? "Ollama" : "Not generated by Ollama"}</strong>
                    <span>Model</span><strong>{message.meta.model}</strong>
                    <span>LLM Used</span><strong>{isVerifiedOllamaResponse(message.meta) ? "Yes" : "No"}</strong>
                    <span>Generated At</span><strong>{isVerifiedOllamaResponse(message.meta) ? formatGeneratedAt(message.meta.generated_at) : "Unavailable"}</strong>
                    <span>Agent</span><strong>{message.meta.selected_agent}</strong>
                    <span>Tools</span><strong>{message.meta.tools_called?.length ? message.meta.tools_called.join(", ") : "None"}</strong>
                    <span>Response Time</span><strong>{formatter.format(message.meta.response_time_ms ?? 0)} ms</strong>
                  </div>
                </details>
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
            {isSending ? "Sending..." : "Ask Inventory Pilot AI Assistant"}
          </button>
        </form>
      </section>
    </div>
  );
}

function ArchitecturePage({ onAsk }) {
  const agentCards = [
    ["Supervisor Agent", "Decides what the user is asking and routes the request to the correct specialist."],
    ["Inventory Agent", "Handles product lookup, stock levels, supplier details, availability, and reorder questions."],
    ["Forecast Agent", "Uses historical sales or movement data and forecasting tools to calculate future demand and reorder recommendations."],
    ["Document Agent", "Uses RAG and document search to answer questions from uploaded or internal documents."],
    ["Report Agent", "Generates inventory, forecast, and operations reports."],
    ["Quality Review Agent", "Checks the answer before it is returned to the user."],
  ];

  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">AI Architecture</p>
            <h3>How the Inventory Pilot AI agents work</h3>
          </div>
          <button className="secondary-button" onClick={onAsk}>
            Open Inventory Pilot AI Assistant
          </button>
        </div>
        <div className="architecture-diagram">
          <div className="diagram-node">User request</div>
          <div className="diagram-arrow">v</div>
          <div className="diagram-node diagram-highlight">Supervisor Agent</div>
          <div className="diagram-arrow">v</div>
          <div className="diagram-grid">
            <div className="diagram-node">Inventory Agent</div>
            <div className="diagram-node">Forecast Agent</div>
            <div className="diagram-node">Document Agent</div>
            <div className="diagram-node">Report Agent</div>
          </div>
          <div className="diagram-arrow">v</div>
          <div className="diagram-node">Quality Review Agent</div>
          <div className="diagram-arrow">v</div>
          <div className="diagram-node diagram-final">Final Response</div>
        </div>
        <div className="supplier-grid architecture-grid">
          {agentCards.map(([title, description]) => (
            <article key={title} className="supplier-card">
              <strong>{title}</strong>
              <p>{description}</p>
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
          <p>The LLM may explain the forecast in plain English, but the actual numbers come from deterministic calculation and stored historical or demo data.</p>
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

function AdminPage({ diagnostics }) {
  const recentRequests = diagnostics?.recent_requests ?? [];
  const aiEvents = diagnostics?.ai_events ?? [];
  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Admin</p>
            <h3>Diagnostics</h3>
          </div>
        </div>
        <div className="supplier-grid admin-summary-grid">
          <article className="supplier-card">
            <strong>Current LLM Provider</strong>
            <p>{diagnostics?.provider ?? "Unknown"}</p>
          </article>
          <article className="supplier-card">
            <strong>Current Model</strong>
            <p>{diagnostics?.model ?? "Unknown"}</p>
          </article>
          <article className="supplier-card">
            <strong>Ollama Base URL</strong>
            <p>{diagnostics?.ollama_base_url ?? "Unknown"}</p>
          </article>
          <article className="supplier-card">
            <strong>Ollama Reachable</strong>
            <p>{diagnostics?.ollama_reachable ? "Yes" : "No"}</p>
          </article>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">Recent requests</p>
            <h3>Last 20 requests</h3>
          </div>
        </div>
        <div className="list-stack">
          {recentRequests.map((trace, index) => (
            <article key={`${trace.session_id}-${index}`} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{trace.selected_agent || "Agent"}</strong>
                  <span>{new Date(trace.created_at).toLocaleString()}</span>
                </div>
                <strong>{formatter.format(trace.response_time_ms ?? 0)} ms</strong>
              </div>
              <p>{trace.user_input}</p>
              <div className="meta-row">
                <span>Provider: {trace.provider}</span>
                <span>Model: {trace.model}</span>
              </div>
            </article>
          ))}
          {!recentRequests.length ? <div className="empty-state">No requests recorded yet.</div> : null}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="section-label">AI Events</p>
            <h3>Recent Ollama generations</h3>
          </div>
        </div>
        <div className="list-stack">
          {aiEvents.map((event) => (
            <article key={event.id} className="list-card">
              <div className="list-card-top">
                <div>
                  <strong>{event.feature_name}</strong>
                  <span>{formatGeneratedAt(event.created_at)}</span>
                </div>
                <strong>{event.status}</strong>
              </div>
              <div className="meta-row">
                <span>Feature: {event.feature_name}</span>
                <span>Provider: {event.provider}</span>
                <span>Model: {event.model}</span>
              </div>
              <div className="meta-row">
                <span>LLM Used: {event.llm_used ? "Yes" : "No"}</span>
                <span>Status: {event.status}</span>
                <span>Error: {event.error_message || "None"}</span>
              </div>
            </article>
          ))}
          {!aiEvents.length ? <div className="empty-state">No AI events recorded yet.</div> : null}
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
  return {
    dashboard: "Dashboard",
    inventory: "Inventory Management",
    products: routeState.params.filter === "high-demand" ? "High Demand Products" : "Products",
    forecasts: "Forecast Alerts",
    reorder: "Reorder Recommendations",
    suppliers: "Suppliers",
    reports: "Reports",
    ask: "Inventory Pilot AI Assistant",
    architecture: "AI Architecture",
    admin: "Admin / Diagnostics",
  }[routeState.route] ?? "Inventory Pilot AI";
}

function insightPath(routeState, selectedProductId) {
  if (routeState.route === "dashboard") {
    return "/ai/insights/dashboard";
  }
  if (routeState.route === "products") {
    const params = new URLSearchParams();
    params.set("filter", routeState.params.filter ?? "all");
    if (selectedProductId) {
      params.set("product_id", String(selectedProductId));
    }
    return `/ai/insights/products?${params.toString()}`;
  }
  if (routeState.route === "forecasts") {
    return "/forecast/ai-explanation";
  }
  if (routeState.route === "reorder") {
    return "/reorder/ai-explanation";
  }
  if (routeState.route === "suppliers") {
    return "/ai/insights/suppliers";
  }
  if (routeState.route === "reports") {
    return "/ai/insights/reports";
  }
  return "";
}
