echo "\033[0;33m:::::: Add src to pythonpath\033[0m"
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
echo "PYTHONPATH: ${PYTHONPATH}"

echo "\033[0;33m:::::: Run Jupyter notebooks\033[0m"
pytest --nbmake examples/ --ignore-glob='examples/*emcee*' --nbmake-timeout=300 --color=yes -n=auto
