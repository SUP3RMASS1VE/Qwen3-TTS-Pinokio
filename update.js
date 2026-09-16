module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull",
      path: "app"
    }
  }, {
    when: "{{platform === 'win32' && gpu === 'amd'}}",
    method: "fs.copy",
    params: {
      src: "app_amd.py",
      dest: "app/app_amd.py"
    }
  }]
}