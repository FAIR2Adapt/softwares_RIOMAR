import process from "node:process";

const GH_TOKEN = process.env.GH_TOKEN;
const OWNER = process.env.OWNER;
const REPO = process.env.REPO;
const PROJECT_TITLE = process.env.PROJECT_TITLE;
const ITEMS = JSON.parse(process.env.ITEMS_JSON || "[]");

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
    console.error(JSON.stringify(json, null, 2));
    throw new Error("GraphQL request failed");
  }
  return json.data;
}

async function getRepoId() {
  const q = `
    query($owner: String!, $repo: String!) {
      repository(owner: $owner, name: $repo) { id }
    }
  `;
  const data = await gql(q, { owner: OWNER, repo: REPO });
  return data.repository.id;
}

async function findProject(repoId, title) {
  const q = `
    query($repoId: ID!) {
      node(id: $repoId) {
        ... on Repository {
          projectsV2(first: 50) { nodes { id title number } }
        }
      }
    }
  `;
  const data = await gql(q, { repoId });
  const nodes = data.node.projectsV2.nodes || [];
  return nodes.find((p) => p.title === title) || null;
}

async function createProject(repoId, title) {
  const m = `
    mutation($ownerId: ID!, $title: String!) {
      createProjectV2(input: { ownerId: $ownerId, title: $title }) {
        projectV2 { id title number }
      }
    }
  `;
  const data = await gql(m, { ownerId: repoId, title });
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
  return data.node.fields.nodes;
}

async function createStatusField(projectId) {
  // SINGLE_SELECT options are specified when creating the field. :contentReference[oaicite:3]{index=3}
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
  if (!GH_TOKEN) throw new Error("Missing GH_TOKEN");
  if (!OWNER || !REPO) throw new Error("Missing OWNER/REPO");

  const repoId = await getRepoId();

  let project = await findProject(repoId, PROJECT_TITLE);
  if (!project) {
    project = await createProject(repoId, PROJECT_TITLE);
    console.log(`Created repo project: ${project.title} (#${project.number})`);
  } else {
    console.log(`Found repo project: ${project.title} (#${project.number})`);
  }

  // Ensure Status field exists
  let fields = await listFields(project.id);
  let status = fields.find((f) => f.__typename === "ProjectV2SingleSelectField" && f.name === "Status");

  if (!status) {
    status = await createStatusField(project.id);
    console.log(`Created Status field`);
  } else {
    console.log(`Status field already exists`);
  }

  const todoOption = status.options.find((o) => o.name === "Todo");
  const fieldId = status.id;

  // Add draft items and set them to Todo
  for (const t of ITEMS) {
    const itemId = await addDraftItem(project.id, t);
    console.log(`Added draft item: ${t}`);
    if (todoOption?.id) {
      await setSingleSelect(project.id, itemId, fieldId, todoOption.id);
      console.log(`  -> Status: Todo`);
    }
  }

  console.log(`PROJECT_ID=${project.id}`);
  console.log(`PROJECT_NUMBER=${project.number}`);
})();
