# Makefile for Pokemon Deep Research Agent
# Compatible with MacBook M3 Pro and other platforms

CONTAINER_NAME?=pokemon-deep-research-agent
PYTHON?=python3
PIP?=pip3

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

.PHONY: help build start stop shell test clean deps install run demo status

help: ## Show this help message
	@echo "$(BLUE)Pokemon Deep Research Agent - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@echo "  build     - Build the Docker image"
	@echo "  start     - Start the development container"
	@echo "  stop      - Stop the development container"
	@echo "  shell     - Open shell in running container"
	@echo ""
	@echo "$(GREEN)Local Development:$(NC)"
	@echo "  install   - Install dependencies locally"
	@echo "  run       - Run the agent locally"
	@echo "  demo      - Run demo locally"
	@echo "  test      - Run tests locally"
	@echo "  status    - Check system status"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  deps      - Update dependencies"
	@echo "  clean     - Clean up containers and images"
	@echo "  help      - Show this help"

build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(NC)"
	@if [ "$$(uname -m)" = "arm64" ]; then \
		echo "$(BLUE)Detected ARM64 (Apple Silicon)$(NC)"; \
		docker build --platform linux/arm64 -t $(CONTAINER_NAME) .; \
	else \
		echo "$(BLUE)Detected AMD64$(NC)"; \
		docker build --platform linux/amd64 -t $(CONTAINER_NAME) .; \
	fi
	@echo "$(GREEN)✅ Build complete$(NC)"

start: ## Start development container
	@echo "$(YELLOW)Starting development container...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(RED)❌ .env file not found. Please copy .env.example to .env and configure.$(NC)"; \
		exit 1; \
	fi
	@docker-compose -f docker-compose.dev.yaml up -d
	@echo "$(GREEN)✅ Container started$(NC)"
	@echo "$(BLUE)Use 'make shell' to access the container$(NC)"

stop: ## Stop development container
	@echo "$(YELLOW)Stopping development container...$(NC)"
	@docker-compose -f docker-compose.dev.yaml down
	@echo "$(GREEN)✅ Container stopped$(NC)"

shell: ## Open shell in running container
	@echo "$(BLUE)Opening shell in container...$(NC)"
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		docker exec -it $(CONTAINER_NAME) /bin/bash; \
	else \
		echo "$(RED)❌ Container not running. Use 'make start' first.$(NC)"; \
		exit 1; \
	fi

install: ## Install dependencies locally
	@echo "$(YELLOW)Installing dependencies locally...$(NC)"
	@if command -v $(PIP) >/dev/null 2>&1; then \
		$(PIP) install -r requirements.txt; \
		echo "$(GREEN)✅ Dependencies installed$(NC)"; \
	else \
		echo "$(RED)❌ pip3 not found. Please install Python 3.11+$(NC)"; \
		exit 1; \
	fi

run: ## Run the agent locally
	@echo "$(BLUE)Starting Pokemon Deep Research Agent...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️  .env file not found, using environment variables$(NC)"; \
	fi
	@$(PYTHON) main.py --interactive

demo: ## Run demo locally
	@echo "$(BLUE)Running Pokemon Deep Research Agent Demo...$(NC)"
	@$(PYTHON) demo.py

test: ## Run tests locally
	@echo "$(YELLOW)Running test suite...$(NC)"
	@$(PYTHON) run_tests.py

status: ## Check system status
	@echo "$(BLUE)Checking system status...$(NC)"
	@$(PYTHON) main.py --status

deps: ## Update dependencies
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	@if [ -f requirements/dev.in ]; then \
		pip-compile requirements/dev.in; \
		pip-compile requirements.txt; \
	else \
		echo "$(BLUE)Using simple requirements.txt$(NC)"; \
	fi
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

clean: ## Clean up Docker containers and images
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	@docker-compose -f docker-compose.dev.yaml down --volumes --remove-orphans
	@if [ "$$(docker images -q $(CONTAINER_NAME))" ]; then \
		docker rmi $(CONTAINER_NAME); \
	fi
	@docker system prune -f
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

# Default target
.DEFAULT_GOAL := help

