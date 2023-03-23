<template>
  <div>
    <div class="header">
      <div>
        <h1>EMOTIONAL SOUNDSCAPE</h1>
        <h2
          class="MainLogo_Description__OjeOd Text_Container___aClu Text_Container-EN__IJw2R"
        >Take a walk in the soundscapes selected by AI</h2>
      </div>
    </div>

    <div>
      <div id="app">
        <el-upload
          action=""
          list-type="picture-card"
          drag
          :auto-upload="false"
          :on-change="handleelchange"
          :on-remove="handleRemove"
          :class="{disabled:this.fileList.length === 1}"
          :file-list="fileList"
          class="uploadBox"
        >
          <i class="el-icon-plus"></i>

        </el-upload>
        <br><br>
        <div class="textBox">
          只能上传jpg等图片格式，且不能超过10MB
        </div>
      </div>
    </div>
  </div>
</template>

<script>
//import axios from "axios"
//import axios from './api/axios'
import axios from 'axios'
export default {
  name: 'App',
  components: {
  },
  data () {
    return {
      // 查看放大图的url
      dialogImageUrl: '',
      // 上传文件数据
      file_data: null,
      // 从后端获取的图片url
      image_url: null,
      // 是否展示获取的图片
      image_show: false,
      // 当前上传的图片数量
      picture_amount: 0,
      fileList: [],
    }
  },
  methods: {
    handleelchange (file, filelist) {
      console.log(file, filelist)
      // axios.get("/API/test", formdata)
      //   .then(res => { console.log(res) })
      //   .catch((err) => {
      //     console.log(err)
      //   })
      this.fileList = filelist;
      this.file_data = file;
      this.picture_amount += 1;
      const isIMAGE = (file.raw.type === 'image/jpeg' || file.raw.type === 'image/png' || file.raw.type === 'image/gif' ||
      file.raw.type == 'image/bmp');
      const isLt1M = file.size / 1024 / 1024 < 10;

      if (!isIMAGE) {
        this.$message.error('上传文件只能是图片格式!');
        this.picture_amount -= 1;
        // 从文件列表中删除最后一个元素
        this.fileList.pop();
        return false;
      }
      if (!isLt1M) {
        this.$message.error('上传文件大小不能超过 10MB!');
        this.picture_amount -= 1;
        // 从文件列表中删除最后一个元素
        this.fileList.pop();
        return false;
      }


      axios.request({
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        url: "/API/test",
        method: "post",
        data: file,
        withCredentials: false,
      })
        .then((res) => {
          console.log(res)
          console.log(file)
        })
        .catch((err) => {
          console.log(err)
          this.$message({
            showClose: true,
            message: '无法识别人脸！',
            type: 'warning'
          });
        })
    },
    handleRemove (file, fileList) {
      setTimeout(() => {
        this.fileList = fileList;
      }, 1000);
    },
  },
  beforeCreate () {
    document
      .querySelector('body')
      .setAttribute('style', 'background-color:#e6ffe6')
  },

}


</script>

<style lang="less" scoped>
html,
body {
  margin: 0;
  padding: 0;
}
#app {
  height: 50vh;
  width: 100%;
  color: #2c3e50;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

page {
  background: #ebecee;
}
.header {
  display: flex;
  justify-content: center; /*垂直居中*/
  align-items: center; /*水平居中*/

  margin-top: 150px;
  // border: 1px solid black;
  align-content: center;
  height: 50px;
  .left_content {
    display: flex;
    flex: 20;
    .tabButton {
      flex: 1;
      height: 30px;
      width: 30px;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .header-logo-box {
      flex: 1000;
      display: flex;
      align-items: center;
      margin-left: 40%;
      .header-logo {
        height: 80%;
      }
      .header-text {
        height: 100%;
      }
    }
  }
  .right_content {
    flex: 1;
    display: flex;
    align-content: center;

    .el-dropdown-link {
      .el-icon-s-custom {
        font-size: 40px;
      }
    }
  }
}

h1 {
  font-size: 60px;
  font-family: "华文行楷";
  color: rgb(115, 230, 192);
}
h2 {
  font-size: 37px;
  font-family: "华文新魏";
  color: rgb(28, 170, 28);
}

.disabled /deep/ .el-upload--picture-card {
  display: none !important;
}

//缩略图正常比例
/deep/.el-upload-list--picture-card .el-upload-list__item {
  height: 300px !important;
  width: 100% !important;

  position: relative;
  left: 120px;
  top: 50px;
}

.uploadBox {
  position: relative;
  right: 120px;
}

.textBox{
  position:relative;
  right:15px;
}
</style>
