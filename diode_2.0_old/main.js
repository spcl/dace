const { app, BrowserWindow } = require('electron')
const ipc = require('electron').ipcMain
const XMLHttpRequest = require('xhr2');

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let win

function renderSDFG(sdfg) {
  win.webContents.send("renderSDFG", sdfg)
}

function OnClickCompile(code) {
  // send message to diode rest api "compile that code"
  var xhr = new XMLHttpRequest();
  var url = "http://localhost:5000/dace/api/v1.0/compile";
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4 && xhr.status === 200) {
      var compile_result = JSON.parse(xhr.responseText);
      renderSDFG(compile_result.sdfg)
    }
  };
  var data = JSON.stringify({ "code": code });
  xhr.send(data);
}


function createMenu() {
  const { Menu, MenuItem } = require('electron')
  const menu = new Menu()

  menu.append(new MenuItem({
    label: 'Open',
    accelerator: 'Ctrl+O',
    click: () => {
      win.webContents.send('openFile')
    }
  }));


  menu.append(new MenuItem({
    label: 'Compile',
    accelerator: 'Ctrl+F5',
    click: () => {
      // send message to render process "give me the current text"
      win.webContents.send('getText')
      ipc.on('getTextResponse', function (event, arg) {
        OnClickCompile(arg)
      })
    }
  }));

  menu.append(new MenuItem({
    label: 'DevConsole',
    accelerator: 'Ctrl+D',
    click: () => {
      win.webContents.openDevTools()
    }
  }))

  Menu.setApplicationMenu(menu);
}

function createWindow() {
  // Create the browser window.
  win = new BrowserWindow({ width: 800, height: 600 })
  win.ELECTRON_DISABLE_SECURITY_WARNINGS = true;

  // and load the index.html of the app.
  win.loadFile('index.html')

  // create Menu
  createMenu()


  // Emitted when the window is closed.
  win.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    win = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (win === null) {
    createWindow()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.

var server_healt_periodic = setInterval(
  function () {
    var xhr = new XMLHttpRequest();
    xhr.timeout = 600;
    var url = "http://localhost:5000/dace/api/v1.0/status";
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4 && xhr.status === 200) {
        win.webContents.send('serverAlive');
        clearInterval(server_healt_periodic);
        console.log("Server is alive!");
      }
    };
    var data = JSON.stringify({ "health": "just_asking" });
    xhr.send(data);
  }, 5000);
