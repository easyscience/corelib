window.MathJax = {
  tex: {
    //inlineMath: [['\\(', '\\)']],
    //displayMath: [['\\[', '\\]']],
    // Add support for $...$ and \(...\) delimiters
    inlineMath: [
      ['$', '$'],
      ['\\(', '\\)'],
    ],
    // Add support for $$...$$ and \[...]\ delimiters
    displayMath: [
      ['$$', '$$'],
      ['\\[', '\\]'],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    //ignoreHtmlClass: ".*|",
    //processHtmlClass: "arithmatex"
    // Skip code blocks only
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    // Only ignore explicit opt-out
    ignoreHtmlClass: 'no-mathjax|tex2jax_ignore',
  },
}

document$.subscribe(() => {
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})
