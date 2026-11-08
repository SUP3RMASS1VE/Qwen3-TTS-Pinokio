module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: [
        "git checkout -- app.py",
        "git pull"
      ],
      path: "app"
    }
  }, {
    method: "shell.run",
    params: {
      message: "python patch_app.py"
    }
  }]
}
