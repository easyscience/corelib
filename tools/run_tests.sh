echo "\033[0;33m:::::: Add src to pythonpath\033[0m"
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
echo "PYTHONPATH: ${PYTHONPATH}"

echo "\033[0;33m:::::: Run unit tests\033[0m"
pytest tests/unit_tests/ --color=yes --disable-warnings

echo "\033[0;33m:::::: Run functional tests\033[0m"
pytest tests/functional_tests/ --color=yes --disable-warnings

echo "\033[0;33m:::::: Run integration tests\033[0m"
pytest tests/integration_tests/ --color=yes --disable-warnings
