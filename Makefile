.PHONY: help db_list db_apply db_init db_seed test_tool test_intents test_intents_verbose test_intents_domain

help:
	@echo "Comandos disponíveis:"
	@echo "  make db_list        - Lista os arquivos .sql detectados (ordem de execução)"
	@echo "  make db_init        - Aplica o schema inicial e o seed de status"
	@echo "  make db_apply       - Aplica todos os arquivos .sql do diretório sql/"
	@echo "  make db_seed        - Reaplica apenas o seed de status"
	@echo "  make db_truncate    - Trunca tabelas do Mini-CRM (pede confirmação)"
	@echo "  make db_truncate_yes- Trunca tabelas do Mini-CRM (sem prompt)"
	@echo "  make test_tools     - Roda os testes de ferramentas do agente"
	@echo "  make test_tools_v   - Roda os testes de ferramentas com prints (pytest -s)"
	@echo "  make test_quick     - Roda o teste rápido de multi-intents (sem LLM/DB)"
	@echo "  make test_multi     - Roda apenas os testes marcados como @slow (multi-intents)"
	@echo "  make test_fast      - Roda apenas testes NÃO marcados como @slow"
	@echo "  make test_pytest    - Roda toda a suíte pytest"
	@echo "  make test_intents   - Roda testes de intents do agente (script)"
	@echo "  make test_intents_v - Roda testes de intents com output detalhado"
	@echo "  make test_intents_domain DOMAIN=leads - Roda testes filtrados por domínio"

# ---- Banco de Dados (migracoes simples via scripts/migrate.py) ----
db_list:
	python scripts/migrate.py --dir sql --list

db_apply:
	python scripts/migrate.py --dir sql

db_init:
	python scripts/migrate.py --dir sql --files 00_drop_crm_tables.sql 01_crm_schema.sql 02_seed_status_lead.sql

db_seed:
	python scripts/migrate.py --dir sql --files 02_seed_status_lead.sql

db_truncate:
	python scripts/db_truncate.py

db_truncate_yes:
	python scripts/db_truncate.py --yes

test_tools:
	python -m pytest -q tests/test_tools_m1c.py

test_tools_v:
	python -m pytest -s -q tests/test_tools_m1c.py

test_quick:
	python -m pytest -q tests/test_quick_multi_intents.py

test_multi:
	python -m pytest -q -m slow tests/test_multi_intents.py

test_fast:
	python -m pytest -q -m "not slow" tests

test_pytest:
	python -m pytest -q tests

# ---- Testes de Intents do Agente ----
test_intents:
	python tests/test_intents.py

test_intents_v:
	python tests/test_intents.py --verbose

test_intents_domain:
	python tests/test_intents.py --domain $(DOMAIN)
