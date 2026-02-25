import process from "node:process";

const GH_TOKEN = process.env.GH_TOKEN;
const OWNER = process.env.OWNER; // org or username
const REPO = process.env.REPO;
const PROJECT_TITLE = process.env.PROJECT_TITLE;
const ITEMS = JSON.parse(process.env.ITEMS_JSON || "[]");

if (!GH_TOKEN) throw new Error("Missing GH_TOKEN");
if (!OWNER) throw new Error("Missing OWNER");
if (!REPO) throw new Error("Missing REPO");
if (!PROJECT_TITLE) throw new Error("Missing PROJECT_TITLE");

async function gql(query, variables) {
  const res = await fetch("https://api.github.com/graphql", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${GH_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, variables }),
  });

  const json = await res.json();
  if (!res.ok || json.errors) {
    console.error("GraphQL error response:", JSON.stringify(json, null, 2));
    throw new Error("GraphQL request failed");
  }
  return json.data;
}

async function getRepoId(owner, repo) {
  const q = `
    query($owner: String!, $repo: String!) {
      repository(owner: $owner, name: $repo) { id }
    }
  `;
  const data = await gql(q, { owner, repo });
  return data.repository.id;
}

async function getOwnerId(login) {
  const q = `
    query($login: String!) {
      organization(login: $login) { id }
      user(login: $login) { id }
    }
  `;
  const data = await gql(q, { login });
  const id = data.organization?.id ?? data.user?.id;
  if (!id) throw new Error(`Could not resolve owner login: ${login}`);
  return id;
}

async function findOwnerProject(ownerLogin, title) {
  // We look under org OR user based on which exists for OWNER
  const q = `
    query($login: String!) {
      organization(login: $login) {
        projectsV2(first: 50) { nodes { id title number } }
      }
      user(login: $login) {
        projectsV2(first: 50) { nodes { id title number } }
      }
    }
  `;
  const data = await gql(q, { login: ownerLogin });
  const projects =
    data.organization?.projectsV2?.nodes ??
    data.user?.projectsV2?.nodes ??
    [];
  return projects.find((p) => p.title === title) || null;
}

async function createProjectLinkedToRepo(ownerId, repoId, title) {
  // Link to repo using repositoryIds so it appears in repo's Projects tab.
  const m = `
    mutation($ownerId: ID!, $title: String!, $repoId: ID!) {
      createProjectV2(input: {
        ownerId: $ownerId,
        title: $title,
        repositoryIds: [$repoId]
      }) {
        projectV2 { id title number }
      }
    }
  `;
  const data = await gql(m, { ownerId, title, repoId });
  return data.createProjectV2.projectV2;
}

async function listFields(projectId) {
  const q = `
    query($projectId: ID!) {
      node(id: $projectId) {
        ... on ProjectV2 {
          fields(first: 50) {
            nodes {
              __typename
              ... on ProjectV2FieldCommon { id name }
              ... on ProjectV2SingleSelectField {
                id
                name
                options { id name }
              }
            }
          }
        }
      }
    }
  `;
  const data = await gql(q, { projectId });
  return data.node.fields.nodes ?? [];
}

async function createStatusField(projectId) {
  const m = `
    mutation($projectId: ID!) {
      createProjectV2Field(input: {
        projectId: $projectId,
        name: "Status",
        dataType: SINGLE_SELECT,
        singleSelectOptions: [
          { name: "Todo" },
          { name: "In progress" },
          { name: "Done" }
        ]
      }) {
        projectV2Field {
          ... on ProjectV2SingleSelectField {
            id
            name
            options { id name }
          }
        }
      }
    }
  `;
  const data = await gql(m, { projectId });
  return data.createProjectV2Field.projectV2Field;
}

async function addDraftItem(projectId, title) {
  const m = `
    mutation($projectId: ID!, $title: String!) {
      addProjectV2DraftIssue(input: { projectId: $projectId, title: $title }) {
        projectItem { id }
      }
    }
  `;
  const data = await gql(m, { projectId, title });
  return data.addProjectV2DraftIssue.projectItem.id;
}

async function setSingleSelect(projectId, itemId, fieldId, optionId) {
  const m = `
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $fieldId,
        value: { singleSelectOptionId: $optionId }
      }) { clientMutationId }
    }
  `;
  await gql(m, { projectId, itemId, fieldId, optionId });
}

(async () => {
  const repoId = await getRepoId(OWNER, REPO);
  const ownerId = await getOwnerId(OWNER);

  let project = await findOwnerProject(OWNER, PROJECT_TITLE);

  if (!project) {
    project = await createProjectLinkedToRepo(ownerId, repoId, PROJECT_TITLE);
    console.log(`Created project: "${project.title}" (#${project.number})`);
  } else {
    console.log(`Found existing project: "${project.title}" (#${project.number})`);
    console.log(
      `Note: if you just created it manually, ensure it's linked to this repo in Project settings.`
    );
  }

  // Ensure Status field exists
  const fields = await listFields(project.id);
  let statusField = fields.find(
    (f) => f.__typename === "ProjectV2SingleSelectField" && f.name === "Status"
  );

  if (!statusField) {
    statusField = await createStatusField(project.id);
    console.log(`Created "Status" field with Todo / In progress / Done`);
  } else {
    console.log(`"Status" field already exists`);
  }

  const todoOption = statusField.options.find((o) => o.name === "Todo");
  if (!todoOption) {
    console.log(`Warning: "Todo" option not found; skipping status set`);
  }

  for (const title of ITEMS) {
    const itemId = await addDraftItem(project.id, title);
    console.log(`Added draft item: ${title}`);

    if (todoOption) {
      await setSingleSelect(project.id, itemId, statusField.id, todoOption.id);
      console.log(`  -> Status set to Todo`);
    }
  }

  console.log(`PROJECT_ID=${project.id}`);
  console.log(`PROJECT_NUMBER=${project.number}`);
})();
