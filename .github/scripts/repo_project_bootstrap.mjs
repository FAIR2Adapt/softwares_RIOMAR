import process from "node:process";

const GH_TOKEN = process.env.GH_TOKEN;

const REPO_OWNER = process.env.REPO_OWNER;
const REPO_NAME = process.env.REPO_NAME;

const ACTOR_LOGIN = process.env.ACTOR_LOGIN;

const PROJECT_TITLE = process.env.PROJECT_TITLE;
const ITEMS = JSON.parse(process.env.ITEMS_JSON || "[]");

if (!GH_TOKEN) throw new Error("Missing GH_TOKEN");
if (!REPO_OWNER) throw new Error("Missing REPO_OWNER");
if (!REPO_NAME) throw new Error("Missing REPO_NAME");
if (!ACTOR_LOGIN) throw new Error("Missing ACTOR_LOGIN");
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

async function getUserId(login) {
  const q = `
    query($login: String!) {
      user(login: $login) { id login }
    }
  `;
  const data = await gql(q, { login });
  if (!data.user?.id) {
    throw new Error(
      `Could not resolve ACTOR_LOGIN "${login}" to a User. ` +
      `If this run is triggered by a bot (e.g., github-actions[bot]) you must choose another owner.`
    );
  }
  return data.user.id;
}

async function findUserProject(userLogin, title) {
  const q = `
    query($login: String!) {
      user(login: $login) {
        projectsV2(first: 50) { nodes { id title number } }
      }
    }
  `;
  const data = await gql(q, { login: userLogin });
  const nodes = data.user?.projectsV2?.nodes ?? [];
  return nodes.find((p) => p.title === title) || null;
}

async function createProjectLinkedToRepo(ownerUserId, repoId, title) {
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
  const data = await gql(m, { ownerId: ownerUserId, title, repoId });
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
  const repoId = await getRepoId(REPO_OWNER, REPO_NAME);
  const actorUserId = await getUserId(ACTOR_LOGIN);

  console.log(`Repo: ${REPO_OWNER}/${REPO_NAME}`);
  console.log(`Project owner (actor): ${ACTOR_LOGIN}`);

  let project = await findUserProject(ACTOR_LOGIN, PROJECT_TITLE);

  if (!project) {
    project = await createProjectLinkedToRepo(actorUserId, repoId, PROJECT_TITLE);
    console.log(`Created project: "${project.title}" (#${project.number})`);
  } else {
    console.log(`Found existing project: "${project.title}" (#${project.number})`);
    console.log(
      `Note: If it doesn't show under the repo's Projects tab, ensure the project is linked to the repo.`
    );
  }

  // Ensure Status field exists
  const fields = await listFields(project.id);
  let statusField = fields.find(
    (f) => f.__typename === "ProjectV2SingleSelectField" && f.name === "Status"
  );

  if (!statusField) {
    statusField = await createStatusField(project.id);
    console.log(`Created "Status" field`);
  } else {
    console.log(`"Status" field already exists`);
  }

  const todoOption = statusField.options.find((o) => o.name === "Todo");
  if (!todoOption) console.log(`Warning: "Todo" option not found; skipping status set`);

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
