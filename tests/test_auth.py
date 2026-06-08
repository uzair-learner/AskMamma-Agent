from fastapi.testclient import TestClient

from api.backend import app
from scripts.seed_data import seed


def setup_module():
    seed()


client = TestClient(app)


def login(username: str, password: str) -> dict[str, str]:
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_login_success_and_me():
    headers = login("admin@example.com", "AdminPass123!")
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_invalid_password():
    response = client.post("/auth/login", json={"username": "admin@example.com", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "invalid_credentials"


def test_missing_token():
    response = client.get("/auth/me")
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "missing_token"


def test_invalid_token():
    response = client.get("/auth/me", headers={"Authorization": "Bearer bad.token.value"})
    assert response.status_code == 401
    assert response.json()["detail"]["error_code"] == "invalid_token"


def test_viewer_blocked_from_admin_and_write():
    headers = login("viewer@example.com", "ViewerPass123!")
    admin_response = client.get("/admin/diagnostics", headers=headers)
    assert admin_response.status_code == 403
    write_response = client.post(
        "/demo/items",
        json={
            "sku": "VIEWER-BLOCK",
            "name": "Viewer Block",
            "category": "Test",
            "price": 1,
            "cost": 1,
            "stock_quantity": 1,
            "reorder_level": 1,
            "reorder_quantity": 1,
            "confirm": True,
        },
        headers=headers,
    )
    assert write_response.status_code == 403


def test_manager_can_generate_report():
    headers = login("manager@example.com", "ManagerPass123!")
    response = client.get("/reports/askmamma", headers=headers)
    assert response.status_code == 200
    assert response.json()["llm_used"] in {True, False}


def test_tenant_isolation_for_products():
    manager_headers = login("manager@example.com", "ManagerPass123!")
    tenant_b_headers = login("tenantb-viewer@example.com", "TenantBPass123!")
    create_response = client.post(
        "/demo/items",
        json={
            "sku": "TENANT-A-ONLY",
            "name": "Tenant A Only",
            "category": "Test",
            "price": 1,
            "cost": 1,
            "stock_quantity": 5,
            "reorder_level": 2,
            "reorder_quantity": 5,
            "confirm": True,
        },
        headers=manager_headers,
    )
    assert create_response.status_code == 200
    tenant_b_products = client.get("/demo/items?search=TENANT-A-ONLY", headers=tenant_b_headers)
    assert tenant_b_products.status_code == 200
    assert tenant_b_products.json() == []


def test_logout_invalidates_session():
    headers = login("admin@example.com", "AdminPass123!")
    logout_response = client.post("/auth/logout", headers=headers)
    assert logout_response.status_code == 200
    me_response = client.get("/auth/me", headers=headers)
    assert me_response.status_code == 401
    assert me_response.json()["detail"]["error_code"] == "invalid_session"


def test_reorder_request_workflow_and_audit_log():
    headers = login("manager@example.com", "ManagerPass123!")
    supplier_response = client.get("/demo/suppliers", headers=headers)
    supplier_id = supplier_response.json()[0]["id"]
    create_response = client.post(
        "/reorder/requests",
        json={
            "supplier_id": supplier_id,
            "status": "Draft",
            "notes": "Demo reorder coordination",
            "items": [{"sku": "OFF-001", "recommended_quantity": 10}],
        },
        headers=headers,
    )
    assert create_response.status_code == 200
    request_id = create_response.json()["id"]
    update_response = client.put(
        f"/reorder/requests/{request_id}",
        json={"status": "Submitted"},
        headers=headers,
    )
    assert update_response.status_code == 200
    assert update_response.json()["status"] == "Submitted"
    audit_response = client.get("/audit/logs", headers=headers)
    assert audit_response.status_code == 200
    actions = {entry["action"] for entry in audit_response.json()}
    assert {"reorder_request_create", "reorder_request_update"}.issubset(actions)


def test_research_question_does_not_crash():
    headers = login("analyst@example.com", "AnalystPass123!")
    response = client.post(
        "/agent/chat",
        json={"message": "Explain the architecture and orchestration for an interview.", "session_id": "research-test"},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["selected_agent"] in {"ResearchAgent", "ReportingAgent"}
