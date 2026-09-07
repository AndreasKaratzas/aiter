# Local workflow actions

Local composite actions share a small sequence of steps inside an existing job. They do not create runners, install candidate packages or replace the full reusable jobs in [workflow sources](../workflow-sources/README.md).

Start with [common actions](common/README.md). Each action documents its checkout and tool prerequisites. Workflows call the local action after checking out the reviewed repository.
