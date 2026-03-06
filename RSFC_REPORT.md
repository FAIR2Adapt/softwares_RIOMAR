# Quality Assessment for softwares_RIOMAR

An automated assessment of the softwares_RIOMAR tool based on the EVERSE software quality indicators, run on 2026-03-06.

## General Information

- **Software:** softwares_RIOMAR
- **Repository:** https://github.com/FAIR2Adapt/softwares_RIOMAR
- **Assessment date:** 2026-03-06T10:58:18Z
- **Total checks:** 41

## Summary

- **Passed (`true`)**: 13
- **Failed (`false`)**: 27
- **Errors (`error`)**: 1

## Results Table

| Test ID | Test Name | Indicator | Output |
|---|---|---|---|
| https://w3id.org/rsfc/test/RSFC-01-1 | There is an identifier and resolves | persistent_and_unique_identifier | false |
| https://w3id.org/rsfc/test/RSFC-01-2 | There is an identifier associated with the software | persistent_and_unique_identifier | false |
| https://w3id.org/rsfc/test/RSFC-01-3 | Software identifier follows a proper schema | persistent_and_unique_identifier | false |
| https://w3id.org/rsfc/test/RSFC-03-1 | Software has releases | has_releases | false |
| https://w3id.org/rsfc/test/RSFC-03-2 | Releases have an id and version number | has_releases | false |
| https://w3id.org/rsfc/test/RSFC-03-3 | Release versions follow a community established convention | versioning_standards_use | false |
| https://w3id.org/rsfc/test/RSFC-03-4 | Release identifiers follow the same scheme | has_releases | false |
| https://w3id.org/rsfc/test/RSFC-03-5 | Last release consistency | has_releases | error |
| https://w3id.org/rsfc/test/RSFC-03-6 | Version number in metadata | descriptive_metadata | false |
| https://w3id.org/rsfc/test/RSFC-04-1 | Metadata exists | descriptive_metadata | false |
| https://w3id.org/rsfc/test/RSFC-04-2 | There is a README | software_has_documentation | true |
| https://w3id.org/rsfc/test/RSFC-04-3 | There are title and description | descriptive_metadata | false |
| https://w3id.org/rsfc/test/RSFC-04-4 | Software has descriptive metadata | descriptive_metadata | false |
| https://w3id.org/rsfc/test/RSFC-04-5 | There is a codemeta file | descriptive_metadata | true |
| https://w3id.org/rsfc/test/RSFC-05-1 | There is a repostatus badge | version_control_use | false |
| https://w3id.org/rsfc/test/RSFC-05-2 | There is contact and/or support metadata | software_has_documentation | false |
| https://w3id.org/rsfc/test/RSFC-05-3 | Software documentation | software_has_documentation | true |
| https://w3id.org/rsfc/test/RSFC-06-1 | Authors are declared | descriptive_metadata | true |
| https://w3id.org/rsfc/test/RSFC-06-2 | Contributors are declared | descriptive_metadata | false |
| https://w3id.org/rsfc/test/RSFC-06-3 | Authors have an ORCID | descriptive_metadata | true |
| https://w3id.org/rsfc/test/RSFC-06-4 | Authors have roles | descriptive_metadata | true |
| https://w3id.org/rsfc/test/RSFC-07-1 | There is an identifier in README or CITATION.cff | persistent_and_unique_identifier | false |
| https://w3id.org/rsfc/test/RSFC-07-2 | Software identifier resolves to software | persistent_and_unique_identifier | false |
| https://w3id.org/rsfc/test/RSFC-08-1 | Metadata record in Software Heritage or Zenodo | archived_in_software_heritage | false |
| https://w3id.org/rsfc/test/RSFC-09-1 | Repository is from Github/Gitlab | version_control_use | true |
| https://w3id.org/rsfc/test/RSFC-12-1 | There is an article citation or reference publication | software_has_citation | false |
| https://w3id.org/rsfc/test/RSFC-13-1 | Dependencies are declared | requirements_specified | false |
| https://w3id.org/rsfc/test/RSFC-13-2 | There are installation instructions | software_has_documentation | true |
| https://w3id.org/rsfc/test/RSFC-13-3 | Dependencies have version numbers | requirements_specified | false |
| https://w3id.org/rsfc/test/RSFC-13-4 | There is a dependencies machine-readable file | requirements_specified | false |
| https://w3id.org/rsfc/test/RSFC-14-1 | Presence of tests in repository | software_has_tests | false |
| https://w3id.org/rsfc/test/RSFC-14-2 | There are actions to automate tests | repository_workflows | false |
| https://w3id.org/rsfc/test/RSFC-15-1 | Software has license | software_has_license | true |
| https://w3id.org/rsfc/test/RSFC-15-2 | License is SPDX compliant | software_has_license | true |
| https://w3id.org/rsfc/test/RSFC-16-1 | License referenced in metadata files | software_has_license | false |
| https://w3id.org/rsfc/test/RSFC-17-1 | Repository active | version_control_use | false |
| https://w3id.org/rsfc/test/RSFC-17-2 | Commit history | version_control_use | true |
| https://w3id.org/rsfc/test/RSFC-17-3 | Commits are linked to issues | version_control_use | false |
| https://w3id.org/rsfc/test/RSFC-18-1 | Repository has citation | software_has_citation | false |
| https://w3id.org/rsfc/test/RSFC-19-1 | Repository has workflows | repository_workflows | true |
| https://w3id.org/rsfc/test/RSFC-20-1 | Repository has an issue tracker | support_issue_tracking | true |

## Detailed Results by Indicator

### archived_in_software_heritage

#### Metadata record in Software Heritage or Zenodo

- **Test ID:** https://w3id.org/rsfc/test/RSFC-08-1
- **Output:** false
- **Process:** Searches for Zenodo and Software Heritage badges in the README file of the repository
- **Evidence:** Could not find neither a Zenodo DOI identifier or a Software Heritage badge in the repository
- **Suggestions:** You should archive your software not only in Github/Gitlab. More information at https://everse.software/RSQKit/archiving_software

### descriptive_metadata

#### Version number in metadata

- **Test ID:** https://w3id.org/rsfc/test/RSFC-03-6
- **Output:** false
- **Process:** Checks if a version number for the software is indicated in the CITATION.cff, codemeta.json or package files(i.e. pyproject.toml, pom.xml, etc.)
- **Evidence:** Could not find a version number for the software in any of the specified files
- **Suggestions:** You should include the version of your software in its metadata. More information at https://everse.software/RSQKit/software_metadata

#### Metadata exists

- **Test ID:** https://w3id.org/rsfc/test/RSFC-04-1
- **Output:** false
- **Process:** Searches for codemeta, citation and package files in the repository
- **Evidence:** Could not find any of the following metadata files: cff, package_file
- **Suggestions:** You should describe your software in metadata files. More information at https://everse.software/RSQKit/software_metadata

#### There are title and description

- **Test ID:** https://w3id.org/rsfc/test/RSFC-04-3
- **Output:** false
- **Process:** Checks if there is a title and a description for the software in the metadata
- **Evidence:** Could not find a description for the project in the repository
- **Suggestions:** Remember to add a description to your software's metadata. More information at https://everse.software/RSQKit/software_metadata

#### Software has descriptive metadata

- **Test ID:** https://w3id.org/rsfc/test/RSFC-04-4
- **Output:** false
- **Process:** Searches for description, programming languages, date of creation and keywords in the repository
- **Evidence:** Could not find any of the following metadata: description, keywords
- **Suggestions:** You should describe your software using metadata. More information at https://everse.software/RSQKit/software_metadata

#### There is a codemeta file

- **Test ID:** https://w3id.org/rsfc/test/RSFC-04-5
- **Output:** true
- **Process:** Searches for a codemeta.json file in the repository
- **Evidence:** A codemeta.json file was found in the root of the repository
- **Suggestions:** No suggestions

#### Authors are declared

- **Test ID:** https://w3id.org/rsfc/test/RSFC-06-1
- **Output:** true
- **Process:** Searches for authors in various files of the repository (i.e. CITATION.cff, AUTHORS.md, codemeta.json)
- **Evidence:** Authors were found in the repository
- **Suggestions:** No suggestions

#### Contributors are declared

- **Test ID:** https://w3id.org/rsfc/test/RSFC-06-2
- **Output:** false
- **Process:** Searches for contributors in various files of the repository (i.e. codemeta.json, pyproject.toml, pom.xml)'
- **Evidence:** Found authors but could not find any contributors in the repository
- **Suggestions:** Your software should also document its contributors if there are any. More information at https://everse.software/RSQKit/documenting_software_project

#### Authors have an ORCID

- **Test ID:** https://w3id.org/rsfc/test/RSFC-06-3
- **Output:** true
- **Process:** Checks if all authors stated in the CITATION.cff file have an ORCID assigned
- **Evidence:** All authors in the codemeta.json file have an orcid identifier
- **Suggestions:** No suggestions

#### Authors have roles

- **Test ID:** https://w3id.org/rsfc/test/RSFC-06-4
- **Output:** true
- **Process:** Checks if all authors stated in a codemeta.json file have a role assigned 
- **Evidence:** All authors defined in the codemeta file have roles assigned
- **Suggestions:** No suggestions

### has_releases

#### Software has releases

- **Test ID:** https://w3id.org/rsfc/test/RSFC-03-1
- **Output:** false
- **Process:** Searches for release tags in the repository
- **Evidence:** Could not find any releases in the repository
- **Suggestions:** You should often launch releases of your software that contain new updates. More information at https://everse.software/RSQKit/releasing_software

#### Releases have an id and version number

- **Test ID:** https://w3id.org/rsfc/test/RSFC-03-2
- **Output:** false
- **Process:** Checks if all of the releases have an identifier and a version
- **Evidence:** Could not find any releases in the repository
- **Suggestions:** You should often launch releases of your software that contain new updates. More information at https://everse.software/RSQKit/releasing_software

#### Release identifiers follow the same scheme

- **Test ID:** https://w3id.org/rsfc/test/RSFC-03-4
- **Output:** false
- **Process:** Checks if all of the version identifiers follow the same scheme
- **Evidence:** Could not find any releases in the repository
- **Suggestions:** You should often launch releases of your software that contain new updates. More information at https://everse.software/RSQKit/releasing_software

#### Last release consistency

- **Test ID:** https://w3id.org/rsfc/test/RSFC-03-5
- **Output:** error
- **Process:** Checks if the latest release tag matches the version stated in the package file of the repository
- **Evidence:** Could not get the necessary information to perform the test, it being releases and/or version in package file
- **Suggestions:** None

### persistent_and_unique_identifier

#### There is an identifier and resolves

- **Test ID:** https://w3id.org/rsfc/test/RSFC-01-1
- **Output:** false
- **Process:** Searches for an identifier (i.e. DOI or SWHID) in the README file of the repository
- **Evidence:** Could not find any identifier in the repository
- **Suggestions:** You should include a resolvable, unique and persistent identifier in your README file. More information at https://everse.software/RSQKit/software_identifiers

#### There is an identifier associated with the software

- **Test ID:** https://w3id.org/rsfc/test/RSFC-01-2
- **Output:** false
- **Process:** Searches for an identifier in the CITATION.cff, codemeta.json and README files
- **Evidence:** Could not find an identifier in any of the CITATION, codemeta or README files
- **Suggestions:** Remember that identifiers should be included in other files aside from README like codemeta.json, CITATION.cff. More information at https://everse.software/RSQKit/software_identifiers

#### Software identifier follows a proper schema

- **Test ID:** https://w3id.org/rsfc/test/RSFC-01-3
- **Output:** false
- **Process:** Checks if the identifiers associated with the software follow any of these schemas: DOI, URN, GITHUB and SWHID
- **Evidence:** Could not find any identifier in the README file
- **Suggestions:** You should include a resolvable, unique and persistent identifier in your README file. More information at https://everse.software/RSQKit/software_identifiers

#### There is an identifier in README or CITATION.cff

- **Test ID:** https://w3id.org/rsfc/test/RSFC-07-1
- **Output:** false
- **Process:** Searches for an identifier in the README or CITATION.cff files of the repository
- **Evidence:** Could not find an identifier in neither of the README or CITATION files in the repository
- **Suggestions:** You should include your software's identifier in your README or CITATION.cff files. More information at 

#### Software identifier resolves to software

- **Test ID:** https://w3id.org/rsfc/test/RSFC-07-2
- **Output:** false
- **Process:** Checks if the identifier found in the README file or metadata files (i.e. codemeta.json, CITATION.cff) resolves to a page that links back to the software repository
- **Evidence:** Could not find any identifier in the repository
- **Suggestions:** You should include a resolvable, unique and persistent identifier in your README file. More information at https://everse.software/RSQKit/software_identifiers

### repository_workflows

#### There are actions to automate tests

- **Test ID:** https://w3id.org/rsfc/test/RSFC-14-2
- **Output:** false
- **Process:** Searches for workflows that contain test or tests in their names
- **Evidence:** Could not find any workflows or actions that mention test in their names
- **Suggestions:** You should include github actions that run tests to ensure quality. More information at https://everse.software/RSQKit/task_automation_github_actions

#### Repository has workflows

- **Test ID:** https://w3id.org/rsfc/test/RSFC-19-1
- **Output:** true
- **Process:** Searches for workflows in the repository
- **Evidence:** Workflows were found in:
	- https://raw.githubusercontent.com/FAIR2Adapt/softwares_RIOMAR/main/.github/workflows/generate-codemeta.yml
	- https://raw.githubusercontent.com/FAIR2Adapt/softwares_RIOMAR/main/.github/workflows/use-rsfc.yml
	- https://raw.githubusercontent.com/FAIR2Adapt/softwares_RIOMAR/main/.github/workflows/software-fairification-pipeline.yml
	- https://raw.githubusercontent.com/FAIR2Adapt/softwares_RIOMAR/main/.github/workflows/fairification-pipeline.yml
	- https://raw.githubusercontent.com/FAIR2Adapt/softwares_RIOMAR/main/.github/workflows/fair_assessment.yml
- **Suggestions:** No suggestions

### requirements_specified

#### Dependencies are declared

- **Test ID:** https://w3id.org/rsfc/test/RSFC-13-1
- **Output:** false
- **Process:** Searches for dependencies in project configuration files, README and dependencies files such as requirements.txt
- **Evidence:** Could not find any dependencies indicated in the repository
- **Suggestions:** You should have your dependencies stated somewhere to enable reproducibility. More information at https://everse.software/RSQKit/reproducible_software_environments

#### Dependencies have version numbers

- **Test ID:** https://w3id.org/rsfc/test/RSFC-13-3
- **Output:** false
- **Process:** Checks if all of the dependencies stated in the machine-readable file (e.g. requirements.txt, pyproject.toml, etc.) of the repository have a version indicated
- **Evidence:** Could not find any dependencies indicated in the repository
- **Suggestions:** You should have your dependencies stated somewhere to enable reproducibility. More information at https://everse.software/RSQKit/reproducible_software_environments

#### There is a dependencies machine-readable file

- **Test ID:** https://w3id.org/rsfc/test/RSFC-13-4
- **Output:** false
- **Process:** Checks if dependencies are indicated in a machine-readable file
- **Evidence:** Could not find any dependencies indicated in the repository
- **Suggestions:** You should have your dependencies stated somewhere to enable reproducibility. More information at https://everse.software/RSQKit/reproducible_software_environments

### software_has_citation

#### There is an article citation or reference publication

- **Test ID:** https://w3id.org/rsfc/test/RSFC-12-1
- **Output:** false
- **Process:** Searches for an article citation or a reference publication in the codemeta and citation files
- **Evidence:** Could not find neither a reference publication or citation to an article in the repository
- **Suggestions:** You should include other forms of citation like article citations and reference publications in your software's metadata. More information at https://everse.software/RSQKit/creating_good_readme

#### Repository has citation

- **Test ID:** https://w3id.org/rsfc/test/RSFC-18-1
- **Output:** false
- **Process:** Searches for a CITATION.cff file and README file in the repository
- **Evidence:** Could not find any citation in the repository
- **Suggestions:** You should include a citation so other people can citate your research software. More information at https://everse.software/RSQKit/citing_software

### software_has_documentation

#### There is a README

- **Test ID:** https://w3id.org/rsfc/test/RSFC-04-2
- **Output:** true
- **Process:** Searches for a README file in the repository
- **Evidence:** There is a README file in the repository
- **Suggestions:** No suggestions

#### There is contact and/or support metadata

- **Test ID:** https://w3id.org/rsfc/test/RSFC-05-2
- **Output:** false
- **Process:** Searches for contact and support information in the repository
- **Evidence:** Could not find any of the following information: contact, support, support_channels
- **Suggestions:** You should include contact information in your software's metadata in case someone wants to ask for information.

#### Software documentation

- **Test ID:** https://w3id.org/rsfc/test/RSFC-05-3
- **Output:** true
- **Process:** Searches for a README file in the root repository and other forms of documentation such as a Read The Docs badge or url
- **Evidence:** Documentation was found in: 	
- https://raw.githubusercontent.com/FAIR2Adapt/softwares_RIOMAR/main/README.md
- **Suggestions:** No suggest

#### There are installation instructions

- **Test ID:** https://w3id.org/rsfc/test/RSFC-13-2
- **Output:** true
- **Process:** Searches for installation instructions in the README file of the repository
- **Evidence:** Installation instructions were found in the repository
- **Suggestions:** No suggestions

### software_has_license

#### Software has license

- **Test ID:** https://w3id.org/rsfc/test/RSFC-15-1
- **Output:** true
- **Process:** Searches for a file named 'LICENSE' or 'LICENSE.md' in the root of the repository.
- **Evidence:** A license was found in:
	- https://raw.githubusercontent.com/FAIR2Adapt/softwares_RIOMAR/main/LICENSE
- **Suggestions:** No suggestions

#### License is SPDX compliant

- **Test ID:** https://w3id.org/rsfc/test/RSFC-15-2
- **Output:** true
- **Process:** Checks if the licenses detected are SPDX compliant
- **Evidence:** Licenses are SPDX compliant
- **Suggestions:** No suggestions

#### License referenced in metadata files

- **Test ID:** https://w3id.org/rsfc/test/RSFC-16-1
- **Output:** false
- **Process:** Searches for licensing information in the codemeta, citation and package files if they exist
- **Evidence:** Could not find any licensing information in the following metadata files: citation, package
- **Suggestions:** Information about your license should be present in other metadata files like codemeta.json, package files or CITATION. More information on https://everse.software/RSQKit/software_metadata

### software_has_tests

#### Presence of tests in repository

- **Test ID:** https://w3id.org/rsfc/test/RSFC-14-1
- **Output:** false
- **Process:** Searches for files and/or directories that mention test in their names
- **Evidence:** Could not find any files or directories that mention test
- **Suggestions:** Your software should include tests to prove its functionability. More information at https://everse.software/RSQKit/testing_software

### support_issue_tracking

#### Repository has an issue tracker

- **Test ID:** https://w3id.org/rsfc/test/RSFC-20-1
- **Output:** true
- **Process:** Checks if there is an issue tracker in the repository.
- **Evidence:** Found an issue tracker in the repository
- **Suggestions:** No suggestions

### version_control_use

#### There is a repostatus badge

- **Test ID:** https://w3id.org/rsfc/test/RSFC-05-1
- **Output:** false
- **Process:** Searches for a repo status badge in the README file of the repository
- **Evidence:** Could not find a repo status badge in the repository
- **Suggestions:** You should include the state of your repository in the README file

#### Repository is from Github/Gitlab

- **Test ID:** https://w3id.org/rsfc/test/RSFC-09-1
- **Output:** true
- **Process:** Checks if the URL provided is indeed a Github or Gitlab repository
- **Evidence:** URL provided is a Github or Gitlab repository
- **Suggestions:** No suggestions

#### Repository active

- **Test ID:** https://w3id.org/rsfc/test/RSFC-17-1
- **Output:** false
- **Process:** Checks if there is a repo_status badge with value Active and if there are commits in the repository
- **Evidence:** Could not find a repo status badge in the repository
- **Suggestions:** You should keep your repository active and indicate it with a repostatus badge

#### Commit history

- **Test ID:** https://w3id.org/rsfc/test/RSFC-17-2
- **Output:** true
- **Process:** Checks if the software repository has a commits history
- **Evidence:** Commits were found in the repository
- **Suggestions:** No suggestions

#### Commits are linked to issues

- **Test ID:** https://w3id.org/rsfc/test/RSFC-17-3
- **Output:** false
- **Process:** Checks if there is at least one of the existing issues (opened or closed) referenced in any of the commits made in the default branch of the repository
- **Evidence:** There is not any commits linked to any issues in the repository
- **Suggestions:** It is good practice to indicate in your commits which issues you are targeting or solving

### versioning_standards_use

#### Release versions follow a community established convention

- **Test ID:** https://w3id.org/rsfc/test/RSFC-03-3
- **Output:** false
- **Process:** Checks if all of the releases versions follow the SemVer or CalVer versioning standards
- **Evidence:** Could not find any releases in the repository
- **Suggestions:** You should often launch releases of your software that contain new updates. More information at https://everse.software/RSQKit/releasing_software
