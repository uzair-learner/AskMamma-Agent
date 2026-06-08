# Security Notes

This project implements demo-grade backend security for a personal technical project.

## Authentication

The API exposes `/auth/login` and `/auth/me`. Passwords are stored with PBKDF2-SHA256 hashes. JWTs are signed with the configured `JWT_SECRET`.

## RBAC

Roles are enforced by FastAPI dependencies:

- `admin`: full access
- `manager`: inventory writes, suppliers, reorder, reports, documents, AI
- `analyst`: read/report/forecast/document/AI access
- `viewer`: read-only inventory and supplier access

Frontend hiding is convenience only. Backend checks are authoritative.

## Tenant Isolation

Authenticated users have a `tenant_id`. Tenant-aware backend queries filter by that tenant so users from one tenant cannot read another tenant's products, traces, reports, AI events, or documents through protected paths.

## AI Failure Handling

Ollama failures return controlled API payloads instead of crashing inventory pages. The ResearchAgent only uses internal project knowledge and local documents.

## Secrets

Do not commit `.env`, real JWT secrets, real passwords, local databases, vector stores, uploads, build output, or virtual environments.
