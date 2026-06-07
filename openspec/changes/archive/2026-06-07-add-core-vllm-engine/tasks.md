# Tasks: add-core-vllm-engine

## 1. Schemas
- [x] 1.1 Add chat request schema
- [x] 1.2 Add chat response schema
- [x] 1.3 Add shared error or message types if needed

## 2. Engine interface
- [x] 2.1 Add base engine interface
- [x] 2.2 Add vLLM engine adapter
- [x] 2.3 Add engine initialization and health-check behavior

## 3. API routes
- [x] 3.1 Add `/v1/chat/completions`
- [x] 3.2 Add `/health`
- [x] 3.3 Wire dependencies cleanly

## 4. Config
- [x] 4.1 Add model config file
- [x] 4.2 Add server config file
- [x] 4.3 Add config loader helper

## 5. Tests and validation
- [x] 5.1 Add unit tests
- [x] 5.2 Add smoke test
- [x] 5.3 Verify the endpoint contract
