/**
 * main.js — Jean Sees the World
 * Vanilla JS, no frameworks, defer-safe
 * Features: todo-list checkbox persistence, FAQ accordion
 */

(function () {
  'use strict';

  /* ----------------------------------------------------------
     Checkbox persistence via localStorage
     Key: "todos:" + page URL pathname
     Value: JSON array of checked input IDs
  ---------------------------------------------------------- */
  function initTodoList() {
    var lists = document.querySelectorAll('.todo-list');
    if (!lists.length) return;

    var storageKey = 'todos:' + window.location.pathname;
    var stored;

    try {
      stored = JSON.parse(localStorage.getItem(storageKey)) || [];
    } catch (e) {
      stored = [];
    }

    lists.forEach(function (list) {
      var checkboxes = list.querySelectorAll('input[type="checkbox"]');

      checkboxes.forEach(function (cb) {
        // Assign a stable ID if not present
        if (!cb.id) {
          var label = cb.closest('label');
          var text = label ? label.textContent.trim().slice(0, 40) : '';
          cb.id = 'todo-' + slugify(text);
        }

        // Restore saved state
        if (stored.indexOf(cb.id) !== -1) {
          cb.checked = true;
          markLabel(cb);
        }

        // Persist on change
        cb.addEventListener('change', function () {
          markLabel(cb);
          var current;
          try {
            current = JSON.parse(localStorage.getItem(storageKey)) || [];
          } catch (e) {
            current = [];
          }
          if (cb.checked) {
            if (current.indexOf(cb.id) === -1) current.push(cb.id);
          } else {
            current = current.filter(function (id) { return id !== cb.id; });
          }
          try {
            localStorage.setItem(storageKey, JSON.stringify(current));
          } catch (e) { /* quota or private mode */ }
        });
      });
    });
  }

  function markLabel(cb) {
    var label = cb.closest('label');
    if (!label) return;
    var span = label.querySelector('span') || label;
    if (cb.checked) {
      label.classList.add('checked');
    } else {
      label.classList.remove('checked');
    }
  }

  function slugify(str) {
    return str
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 40);
  }

  /* ----------------------------------------------------------
     FAQ Accordion
     Uses native <details>/<summary> — JS adds keyboard nav
     and ARIA. CSS handles open/close visuals.
  ---------------------------------------------------------- */
  function initFaqAccordion() {
    var items = document.querySelectorAll('.faq-item');
    if (!items.length) return;

    items.forEach(function (item) {
      var summary = item.querySelector('summary');
      if (!summary) return;

      // Ensure proper ARIA
      summary.setAttribute('role', 'button');
      summary.setAttribute('aria-expanded', item.hasAttribute('open') ? 'true' : 'false');

      item.addEventListener('toggle', function () {
        summary.setAttribute('aria-expanded', item.open ? 'true' : 'false');
      });

      // Optional: close other open items when one opens (accordion behaviour)
      summary.addEventListener('click', function () {
        items.forEach(function (other) {
          if (other !== item && other.open) {
            other.removeAttribute('open');
            var otherSummary = other.querySelector('summary');
            if (otherSummary) otherSummary.setAttribute('aria-expanded', 'false');
          }
        });
      });
    });
  }

  /* ----------------------------------------------------------
     Initialise on DOM ready
  ---------------------------------------------------------- */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  function init() {
    initTodoList();
    initFaqAccordion();
  }

})();
