# AskMamma Interview Guide

## What is an AI Agent?
An AI agent is a system that can receive a goal, decide what step to take next, use tools, and return a result. In this project the agent is not just a chat box. It can route work, call inventory and retrieval tools, and keep memory.

## What is LangChain?
LangChain is the application layer around language models. It gives us prompt templates, tool calling, output parsing, and model wrappers. In this project it is the easiest way to define tools and optionally run tool-calling agents.

## What is LangGraph?
LangGraph is the workflow engine. It lets us build a graph where each node is an agent or step, and each edge decides what happens next. Here it routes from `SupervisorAgent` to a specialist, then to `ReportingAgent`, then to `QualityReviewAgent`.

## What is RAG?
RAG means Retrieval-Augmented Generation. Instead of letting the model guess, we retrieve relevant document chunks first and then answer from that evidence. This project uses chunking, embeddings, FAISS, and semantic search over local files.

## What is MCP?
MCP stands for Model Context Protocol. It is a standard way to expose tools, resources, and prompts to AI systems. This project shows that idea with `/mcp/tools`, `/mcp/resources`, `/mcp/prompts`, and a JSON-RPC endpoint.

## What is A2A?
A2A means agent-to-agent communication. It is about one agent publishing what it can do and accepting structured tasks from another agent. This project demonstrates that with an agent card and `POST /agent/tasks`.

## What is TensorFlow?
TensorFlow is a machine learning framework commonly used for building and training neural networks. In this project it is included as a learning example for demand prediction, not as a required runtime dependency.

## What is PyTorch?
PyTorch is another machine learning framework. It is popular because it feels very Pythonic and is strong for experimentation. This project includes a parallel PyTorch example so learners can compare it with TensorFlow.

## Why use multiple agents?
Multiple agents make responsibilities easier to explain. The supervisor routes work, specialists focus on one domain, the reporting agent packages results, and the quality reviewer checks the final output. That separation makes the architecture easier to maintain and easier to discuss in interviews.

## Project architecture
The user can talk to Streamlit or the FastAPI backend. The backend calls the LangGraph workflow. The supervisor chooses a specialist agent. Tools read inventory or document data. Memory stores conversation, semantic context, and audit traces. MCP and A2A endpoints expose the project as a protocol-friendly agent system.

## End-to-end workflow
1. The user sends a question.
2. `SupervisorAgent` classifies the request.
3. A specialist agent handles inventory, forecasting, documents, research, or reporting.
4. `ReportingAgent` creates a structured summary in Markdown, TXT, and JSON form.
5. `QualityReviewAgent` checks clarity and grounding.
6. The final answer is returned and the trace is stored.
