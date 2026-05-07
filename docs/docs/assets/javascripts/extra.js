;(function () {
  'use strict'

  // Variables
  const header = document.getElementsByTagName('header')[0]

  console.log(window.pageYOffset)

  // Hide-show header shadow
  function toggleHeaderShadow() {
    if (window.pageYOffset <= 0) {
      header.classList.remove('md-header--shadow')
    } else {
      header.classList.add('md-header--shadow')
    }
  }

  // Onload
  window.onload = function () {
    toggleHeaderShadow()
  }

  // Onscroll
  window.onscroll = function () {
    toggleHeaderShadow()
  }
})()
