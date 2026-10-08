# Git and GitHub for a first-time user

## First: Git is not GitHub

**Git** is the history system on your computer. This repository is already useful without an account or internet connection.

**GitHub** is a website that can host a copy of a Git repository so other people can see or collaborate on it.

Think of Git as the save-history mechanism and GitHub as an optional public bookshelf.

## Useful local commands

```powershell
git status
git log --oneline --decorate -n 10
git show --stat
git tag
```

These commands do not publish anything.

## Publishing later

1. Create a GitHub account at <https://github.com/>.
2. Enable two-factor authentication.
3. Decide whether the repository should be public or private.
4. Create an empty repository on GitHub. Do not add a README or license there; this local repository already has its history.
5. GitHub will show a repository URL. Add it locally:

```powershell
git remote add origin https://github.com/YOUR-NAME/gum.git
git branch -M main
git push -u origin main
git push origin --tags
```

Nothing is published until the `git push` command succeeds.

## Before making it public

- choose a software license;
- replace project-author placeholders in `CITATION.cff`;
- scan the commit for private paths, secrets, personal data, and unnecessarily large files;
- ask an uninvolved person to follow `docs/REPRODUCING.md`;
- decide whether to publish the full raw evidence archives or attach them to a release.

GitHub can be added later without changing the scientific freeze: the commit hash and tag identify the frozen local snapshot.

