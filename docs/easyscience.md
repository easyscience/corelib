# The EasyScience framework

EasyScience is a framework of software tools that can be used to build
experimental data analysis packages. For example, it has already been used in
the development of [EasyDiffraction] and [EasyReflectometry]. Two more packages
are about to be started: EasyImaging (Bragg edge imaging) and EasyDynamics
(Quasielastic neutron scattering, QENS).

The framework consists of both front- and back-end elements, known as [EasyApp]
and [EasyScience], respectively. The front-end provides a shared library of
graphical interface elements that can be used to build a graphical user
interface. The back-end offers a toolset to perform model-dependent analysis,
including the ability to plug-in existing calculation engines.

Below is a diagram illustrating the relationship between the modules of the
EasyScience framework:

<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 500" xmlns:bx="https://boxy-svg.com">
  <g data-id="EasyDiffractionBkg" class="semi-hidden">
    <title>EasyDiffractionBkg</title>
    <text style="white-space: pre; fill: rgb(255, 255, 255); font-size: 25.8px;" x="760" y="214.463">Diffraction</text>
    <rect x="20" y="230" width="860" height="90" style="stroke: rgb(0, 0, 0); fill: rgb(16, 74, 102);"/>
  </g>
  <g data-id="EasyDiffraction" class="semi-hidden">
    <title>EasyDiffraction</title>
    <g>
      <rect x="100" y="250" width="240" height="50" style="fill: rgb(164, 209, 114);"/>
      <text style="fill: rgb(51, 51, 51); font-size: 20px; white-space: pre;" x="136" y="280.532">EasyDiffractionLib</text>
    </g>
    <g>
      <rect x="620" y="250" width="240" height="50" style="fill: rgb(96, 188, 232);"/>
      <text style="fill: rgb(51, 51, 51); font-size: 20px; white-space: pre;" x="653.305" y="281.995">EasyDiffractionApp</text>
    </g>
    <path d="M 340.5 272.497 H 554.489 L 554.489 264.997 L 574.489 274.997 L 554.489 284.997 L 554.489 277.497 H 340.5 V 272.497 Z" style="fill: rgb(164, 209, 114);" bx:shape="arrow 340.5 264.997 233.989 20 5 20 0 1@febe93e4"/>
  </g>
  <g data-id="TechniqueIndependantBkg">
    <title>TechniqueIndependantBkg</title>
    <text style="white-space: pre; fill: rgb(255, 255, 255); font-size: 25.8px;" x="792" y="53.532">Generic</text>
    <rect x="20" y="70" width="860" height="90" style="stroke: rgb(0, 0, 0); fill: rgb(133, 87, 29);"/>
  </g>
  <g data-id="EasyBackend">
    <title>EasyBackend</title>
    <g>
      <rect x="40" y="90" width="220" height="50" style="fill: rgb(164, 209, 114);"/>
      <text style="fill: rgb(51, 51, 51); font-size: 20px; white-space: pre;" x="105.196" y="120.995">EasyCore</text>
    </g>
    <g transform="matrix(1, 0, 0, 1, 20, 0)" class="semi-hidden">
      <rect x="280" y="90" width="220" height="50" style="fill: rgb(164, 209, 114);"/>
      <text style="fill: rgb(51, 51, 51); font-size: 20px; white-space: pre;" x="296.614" y="121.995">EasyCrystallography</text>
    </g>
    <g>
      <rect x="55" y="140.5" width="5" height="137" style="fill: rgb(164, 209, 114);"/>
      <path d="M 60 272.5 H 80 L 80 265 L 100 275 L 80 285 L 80 277.5 H 60 V 272.5 Z" style="fill: rgb(164, 209, 114);" bx:shape="arrow 60 265 40 20 5 20 0 1@2faa7cd5" class="semi-hidden"/>
    </g>
    <path d="M 872 171.823 H 961 L 961 164.323 L 981 174.323 L 961 184.323 L 961 176.823 H 872 V 171.823 Z" style="fill: rgb(164, 209, 114);" transform="matrix(0, 1, -1, 0, 491.345825, -731.5)" bx:shape="arrow 872 164.323 109 20 5 20 0 1@2be15923" class="semi-hidden"/>
    <path d="M 260.5 112.5 H 280.5 L 280.5 105 L 300.5 115 L 280.5 125 L 280.5 117.5 H 260.5 V 112.5 Z" style="fill: rgb(164, 209, 114);" bx:shape="arrow 260.5 105 40 20 5 20 0 1@189d618a" class="semi-hidden"/>
  </g>
  <g data-id="EasyFrontend">
    <title>EasyFrontend</title>
    <g>
      <rect x="560" y="90" width="220" height="50" style="fill: rgb(96, 188, 232);"/>
      <text style="fill: rgb(51, 51, 51); font-size: 20px; white-space: pre;" x="630" y="120.995">EasyApp</text>
    </g>
    <g transform="matrix(1, 0, 0, 1, 520, 0)">
      <rect x="55" y="140.5" width="5" height="137" style="fill: rgb(96, 188, 232);"/>
      <path d="M 60 272.5 H 80 L 80 265 L 100 275 L 80 285 L 80 277.5 H 60 V 272.5 Z" style="fill: rgb(96, 188, 232);" bx:shape="arrow 60 265 40 20 5 20 0 1@2faa7cd5" class="semi-hidden"/>
    </g>
  </g>
  <g data-id="EasyReflectometryBkg">
    <title>EasyReflectometryBkg</title>
    <text style="white-space: pre; fill: rgb(255, 255, 255); font-size: 25.8px;" x="720" y="373.532">Reflectometry</text>
    <rect x="20" y="390" width="860" height="90" style="stroke: rgb(0, 0, 0); fill: rgb(73, 111, 29);"/>
  </g>
  <g data-id="EasyReflectometry">
    <title>EasyReflectometry</title>
    <g>
      <rect x="100" y="410" width="240" height="50" style="fill: rgb(164, 209, 114);"/>
      <text style="fill: rgb(51, 51, 51); font-size: 20px; white-space: pre;" x="125.504" y="440.995">EasyReflectometryLib</text>
    </g>
    <g>
      <rect x="620" y="410" width="240" height="50" style="fill: rgb(96, 188, 232);"/>
      <text style="fill: rgb(51, 51, 51); font-size: 20px; white-space: pre;" x="640" y="440.792">EasyReflectometryApp</text>
    </g>
    <g>
      <rect x="55" y="277.5" width="5" height="160" style="fill: rgb(164, 209, 114);"/>
      <path d="M 60 432.5 H 80 L 80 425 L 100 435 L 80 445 L 80 437.5 H 60 V 432.5 Z" style="fill: rgb(164, 209, 114);" bx:shape="arrow 60 425 40 20 5 20 0 1@c3e2b95f"/>
    </g>
    <g transform="matrix(1, 0, 0, 1, 520, 0)">
      <rect x="55" y="277.5" width="5" height="160" style="fill: rgb(96, 188, 232);"/>
      <path d="M 60 432.5 H 80 L 80 425 L 100 435 L 80 445 L 80 437.5 H 60 V 432.5 Z" style="fill: rgb(96, 188, 232);" bx:shape="arrow 60 425 40 20 5 20 0 1@c3e2b95f"/>
    </g>
    <path d="M 340.5 432.5 H 554.489 L 554.489 425 L 574.489 435 L 554.489 445 L 554.489 437.5 H 340.5 V 432.5 Z" style="fill: rgb(164, 209, 114);" bx:shape="arrow 340.5 425 233.989 20 5 20 0 1@a4d804f9"/>
  </g>
</svg>

<!-- prettier-ignore-start -->
[EasyDiffraction]: https://easydiffraction.org
[EasyReflectometry]: https://easyreflectometry.org
[EasyApp]: https://github.com/easyscience/easyapp
[EasyScience]: https://github.com/easyscience/easyscience
<!-- prettier-ignore-end -->
