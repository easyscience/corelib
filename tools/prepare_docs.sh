echo "\033[0;33m:::::: Add the extra files from the ../assets-docs\033[0m"
cp -R ../assets-docs/docs/assets/ docs/assets/
cp -R ../assets-docs/includes/ includes/
cp -R ../assets-docs/overrides/ overrides/

echo "\033[0;33m:::::: Add the extra files from the ../assets-branding\033[0m"
mkdir -p docs/assets/images/
cp ../assets-branding/easydiffraction/logos/dark.svg docs/assets/images/ed-logo_dark.svg
cp ../assets-branding/easydiffraction/logos/light.svg docs/assets/images/ed-logo_light.svg
cp ../assets-branding/easyscience/logos/dark.svg docs/assets/images/logo_dark.svg
cp ../assets-branding/easyscience/logos/light.svg docs/assets/images/logo_light.svg
cp ../assets-branding/easydiffraction/icons/ed-icon_256x256.png docs/assets/images/favicon.png
mkdir -p overrides/.icons/
cp ../assets-branding/easydiffraction/icons/ed-icon_bw.svg overrides/.icons/easydiffraction.svg
cp ../assets-branding/easyscience-org/icons/eso-icon_bw.svg overrides/.icons/easyscience.svg

echo "\033[0;33m:::::: Create the mkdocs.yml configuration file\033[0m"
cat ../assets-docs/mkdocs.yml docs/mkdocs.yml > mkdocs.yml
