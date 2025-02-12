echo "\033[0;33m:::::: Install Python dependencies\033[0m"
pip install '.[dev,charts,docs]'

echo "\033[0;33m:::::: Install npm dependencies\033[0m"
npm install prettier prettier-plugin-toml --save-dev --save-exact
