/**
 * A helper function to load scripts when the DOM is loaded.
 * This waits for everything to be on the page first before running, since
 * some functionality doesn't behave properly until everything is ready.
 */
var sbRunWhenDOMLoaded = (cb) => {
  if (document.readyState != "loading") {
    cb();
  } else if (document.addEventListener) {
    document.addEventListener("DOMContentLoaded", cb);
  } else {
    document.attachEvent("onreadystatechange", function () {
      if (document.readyState == "complete") cb();
    });
  }
};

// Legacy:
// Force the ToC to hide to not overlap the title
// See https://github.com/executablebooks/sphinx-book-theme/blob/master/src/sphinx_book_theme/assets/scripts/index.js
sbRunWhenDOMLoaded(() => {
  document.querySelector("div.bd-sidebar-secondary").classList.add("hide");
});
