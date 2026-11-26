# Hosting on Railway

This deploys the dashboard to Railway from the GitHub repository, for example
`https://wildfire-production.up.railway.app/dashboard/`. It fits the Free plan.
For a self-managed server, see [../README.md](../README.md) instead.

Railway builds `deploy/railway/Dockerfile` (selected by `railway.json` in the
repository root). Railway has no bind mounts, so the forecast files the server
reads are committed in `deploy/railway/serving/` (about 15 MB) and copied into
the image. Training still happens on your own computer.

## 1. Prepare and push the forecast bundle

From the project root on your computer:

```bash
python deploy/railway/build_bundle.py
git add railway.json deploy/railway
git commit -m "Update Railway forecast bundle"
git push
```

The script checks that the forecast files are complete before replacing the
bundle, and copies only what the web server reads.

## 2. Create the Railway service

1. Sign in at [railway.com](https://railway.com) with GitHub.
2. Select **New Project** > **Deploy from GitHub repo**, and pick this
   repository. Allow Railway's GitHub app to access it if asked.
3. Open the service, go to **Settings** > **Source**, and set the branch to the
   one you push to (for example `version-3`). Railway otherwise uses the
   repository's default branch.
4. Go to **Variables** > **Raw Editor**, paste the contents of
   [railway.env.example](railway.env.example), and replace
   `YOUR_FIRMS_API_KEY_HERE` with your NASA FIRMS MAP_KEY. Leave the key out if
   you only want forecasts; the Live Fires tab then reports a missing key.
5. Go to **Settings** > **Networking** > **Generate Domain**, and enter port
   `8000`.
6. Wait for the deployment to show **Active**, then open
   `https://<your-domain>/dashboard/`.

Railway waits for `/metadata` to answer before switching traffic to a new
deployment, so a broken bundle never replaces a working site.

## 3. Update forecasts

After a new training run, repeat step 1. Each push to the selected branch
redeploys automatically.

## Free plan limits

Checked against Railway's documentation in November 2026; confirm current
limits on Railway's pricing page.

- **Memory:** the Free plan allows 0.5 GB. The app peaked at about 230 MB in
  testing with every dashboard endpoint loaded.
- **Credit:** $1 of usage per month after the 30-day trial ($5 one-time
  credit). If it runs out, the service stops until the next month or an upgrade
  to Hobby.
- **Sleeping:** idle Free services sleep. The first visit after a quiet period
  takes several seconds while the container starts, and Live Fires refreshes
  from FIRMS on that start, then every 15 minutes while awake.
- **One replica:** the app keeps its FIRMS cache in memory, so it must run as a
  single instance. `railway.json` sets `numReplicas` to 1.

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails at `COPY deploy/railway/serving` | The bundle was not committed. Run step 1. |
| Deployment fails its health check | Open **Deployments** > **View logs**. A `503` on `/metadata` means the bundle is incomplete; rebuild it. |
| `Application failed to respond` on the domain | The domain port differs from `PORT`. Set both to `8000`. |
| Live Fires shows a missing key | Set `FIRMS_API_KEY` in **Variables**; Railway redeploys automatically. |
| Old forecast still shown after a push | Check **Settings** > **Source** uses the branch you pushed. |
