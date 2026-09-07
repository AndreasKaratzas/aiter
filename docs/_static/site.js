/* SPDX-License-Identifier: MIT */
(() => {
  "use strict";
  const button = (label, action) => {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    element.addEventListener("click", action);
    return element;
  };
  let copyNumber = 0;
  function prepareTables() {
    document.querySelectorAll("table.docutils").forEach(table => {
      const headings = table.tHead?.rows;
      if (headings?.length !== 1 || headings[0].cells.length < 3) return;
      const labels = [...headings[0].cells].map(cell => cell.textContent.trim().replace(/\s+/g, " "));
      const rows = [...table.tBodies].flatMap(body => [...body.rows]);
      if (!rows.length || !labels.every(Boolean) || [...table.rows].some(row =>
        row.cells.length !== labels.length || [...row.cells].some(cell => cell.colSpan !== 1 || cell.rowSpan !== 1)
      )) return;
      table.classList.add("mobile-stacked");
      table.setAttribute("role", "table");
      [...table.rows].forEach(row => row.setAttribute("role", "row"));
      [...headings[0].cells].forEach(cell => cell.setAttribute("role", "columnheader"));
      rows.forEach(row => [...row.cells].forEach((cell, index) => {
        cell.dataset.label = labels[index];
        cell.setAttribute("role", "cell");
      }));
    });
  }
  function copyDiagram(svg) {
    const copy = svg.cloneNode(true);
    const suffix = `-copy-${++copyNumber}`;
    const ids = new Map();
    [copy, ...copy.querySelectorAll("[id]")].forEach(element => {
      if (element.id) ids.set(element.id, element.id + suffix);
    });
    const rewrite = value => value.replace(/#([\w-]+)/g, (match, id) => ids.has(id) ? `#${ids.get(id)}` : match);
    [copy, ...copy.querySelectorAll("*")].forEach(element => {
      for (const attribute of [...element.attributes]) {
        if (attribute.name === "id") element.setAttribute("id", ids.get(attribute.value));
        else element.setAttribute(attribute.name, rewrite(attribute.value));
      }
      if (element.tagName.toLowerCase() === "style") element.textContent = rewrite(element.textContent);
    });
    return copy;
  }
  function expand(svg, opener) {
    const dialog = document.createElement("dialog");
    dialog.className = "diagram-dialog";
    dialog.setAttribute("aria-label", "Expanded diagram");
    const header = document.createElement("div");
    header.className = "diagram-dialog-header";
    const title = document.createElement("strong");
    title.textContent = "Diagram · scroll to explore";
    const controls = document.createElement("div");
    controls.className = "diagram-dialog-controls";
    const canvas = document.createElement("div");
    canvas.className = "diagram-dialog-canvas";
    canvas.tabIndex = 0;
    const copy = copyDiagram(svg);
    const width = svg.viewBox.baseVal.width || 900;
    let scale = 1;
    const zoom = (factor) => {
      scale = Math.min(4, Math.max(.1, scale * factor));
      copy.style.width = `${width * scale}px`;
      copy.style.height = "auto";
    };
    zoom(1);
    controls.append(
      button("−", () => zoom(1 / 1.25)),
      button("+", () => zoom(1.25)),
      button("Fit", () => { scale = 1; zoom(Math.min(1, (canvas.clientWidth - 48) / width)); }),
      button("100%", () => { scale = 1; zoom(1); }),
      button("Close", () => dialog.close())
    );
    controls.children[0].setAttribute("aria-label", "Zoom out");
    controls.children[1].setAttribute("aria-label", "Zoom in");
    header.append(title, controls);
    canvas.append(copy);
    dialog.append(header, canvas);
    document.body.append(dialog);
    dialog.addEventListener("close", () => { dialog.remove(); opener.focus(); });
    dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
    dialog.showModal();
    controls.lastChild.focus();
    const firstNode = copy.querySelector("g.node.default");
    if (firstNode) {
      const nodeBox = firstNode.getBoundingClientRect();
      const canvasBox = canvas.getBoundingClientRect();
      canvas.scrollLeft += nodeBox.left - canvasBox.left - (canvas.clientWidth - nodeBox.width) / 2;
      canvas.scrollTop += nodeBox.top - canvasBox.top - (canvas.clientHeight - nodeBox.height) / 2;
    }
  }
  function download(svg) {
    const content = new XMLSerializer().serializeToString(svg);
    const url = URL.createObjectURL(new Blob([content], {type: "image/svg+xml"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = "aiter-diagram.svg";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function ready() {
    prepareTables();
    document.querySelectorAll("div.highlight").forEach(element => { element.tabIndex = 0; });
    const diagrams = document.querySelectorAll(".mermaid");
    if (!diagrams.length) { document.documentElement.dataset.diagrams = "ready"; return; }
    try {
      mermaid.initialize({
        startOnLoad: false, securityLevel: "strict", theme: "base",
        fontFamily: "Arial, sans-serif",
        themeVariables: {primaryColor: "#e8f4f5", primaryTextColor: "#183641", primaryBorderColor: "#679daa", lineColor: "#4a7280", secondaryColor: "#f2f7fa", tertiaryColor: "#fff5df", fontSize: "15px"},
        flowchart: {htmlLabels: false, useMaxWidth: true, curve: "basis"},
        sequence: {useMaxWidth: true},
      });
      await mermaid.run({nodes: diagrams});
      diagrams.forEach(element => {
        const figure = element.closest("figure");
        const svg = element.querySelector("svg");
        if (!svg) throw new Error("Diagram did not produce an SVG");
        svg.style.setProperty("--diagram-min-width", `${Math.min(svg.viewBox.baseVal.width, 560)}px`);
        const viewport = figure.querySelector(".diagram-viewport");
        viewport.tabIndex = 0;
        viewport.setAttribute("aria-label", "Diagram; scroll or choose Expand");
        const open = button("Expand", () => expand(svg, open));
        open.setAttribute("aria-label", "Expand diagram");
        figure.querySelector(".diagram-actions").append(open, button("SVG", () => download(svg)));
      });
      document.documentElement.dataset.diagrams = "ready";
    } catch (error) {
      document.documentElement.dataset.diagrams = "error";
      console.error("AITER diagram rendering failed", error);
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ready);
  else ready();
})();
