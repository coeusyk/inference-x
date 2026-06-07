# Tasks: add-core-vllm-engine

## 1. Schemas
- [ ] 1.1 Add chat request schema
- [ ] 1.2 Add chat response schema
- [ ] 1.3 Add shared error or message types if needed

## 2. Engine interface
- [ ] 2.1 Add base engine interface
- [ ] 2.2 Add vLLM engine adapter
- [ ] 2.3 Add engine initialization and health-check behavior

## 3. API routes
- [ ] 3.1 Add `/v1/chat/completions`
- [ ] 3.2 Add `/health`
- [ ] 3.3 Wire dependencies cleanly

## 4. Config
- [ ] 4.1 Add model config file
- [ ] 4.2 Add server config file
- [ ] 4.3 Add config loader helper

## 5. Tests and validation
- [ ] 5.1 Add unit tests
- [ ] 5.2 Add smoke test
- [ ] 5.3 Verify the endpoint contract