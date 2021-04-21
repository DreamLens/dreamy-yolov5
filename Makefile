
.PHONY: all test lint bird-example street-example cuda-example ci-init ci-lint ci-test

data/yolov5:
	@$(shell ./getModels.sh)

# Retrieves yolov5 models
models: | data/yolov5
	@

# Runs lint
lint:
	@echo Linting...
	@golangci-lint  -v --concurrency=3 --config=.golangci.yml --issues-exit-code=1 run \
	--out-format=colored-line-number 

# Run tests
test:
	@mkdir -p reports
	@go test -coverprofile=reports/codecoverage_all.cov ./... -cover -race -p=4
	@go tool cover -func=reports/codecoverage_all.cov > reports/functioncoverage.out
	@go tool cover -html=reports/codecoverage_all.cov -o reports/coverage.html
	@echo "View report at $(PWD)/reports/coverage.html"
	@tail -n 1 reports/functioncoverage.out 