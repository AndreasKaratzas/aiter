# Review and publish the website

The documentation workflow, [host-docs.yml](../.github/workflows/host-docs.yml), builds the site, checks it in Chromium and publishes the checked HTML through GitHub Pages. A pull request produces downloadable HTML and browser evidence for review. Publication runs only after both the build and browser checks pass.

The HTML artifact and browser evidence serve different purposes. HTML is the site to preview. The browser report and screenshots show which pages and interactions were checked, including failures. Retain both when reviewing a documentation change.

Source pages rename hidden repository directories in their public URLs so artifact uploads retain them. For example, a workflow page uses `source/~dot-github/workflows/` while its heading still shows `.github/workflows/`. Downloads contain the original file bytes. Link validation rejects hidden targets even when they exist locally, and a regression test checks source links after applying the Pages action's archive exclusions.

## Local review

Use the commands in [the website guide](README.md). Open the served site, follow the architecture and framework reading paths, expand the class diagram and inspect phone-width screenshots. A Sphinx success message alone does not establish that a table or diagram is readable.

## Remote activation

In the repository's **Settings → Pages**, select **GitHub Actions** as the build source. Enable Actions in a newly created fork, and allow the publishing branch in the `github-pages` environment's deployment rules. The workflow accepts `main`, `docs-website` and the implementation branch `akaratza_aiter_implementation`. A manual run must select one of these branches. For another branch, change both the push filter and the deployment condition together.

The build job has read access to repository contents. The separate deployment job receives `pages: write` and `id-token: write` so GitHub can verify the originating repository, workflow and branch. It publishes the Pages artifact from that same successful run; it does not create a separate source branch or rewrite repository history. Local preview commands do not change repository settings or publish anything.

A project site is served under `https://OWNER.github.io/REPOSITORY/`. Check that address after deployment, including a nested guide, search and phone-width navigation. The site uses relative asset and guide links so the repository prefix is preserved. A successful local build establishes local rendering; the deployment result and a visit to the live URL establish publication.

The workflow definition is the source of truth for deployment conditions and artifact names. A failed build or browser check must remain a failed job. Publishing an earlier artifact while ignoring a failed check would make the public site diverge from its review evidence.
