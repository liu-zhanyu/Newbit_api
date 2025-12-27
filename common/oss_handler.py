"""
OSS文件处理器
提供阿里云OSS的文件上传、下载、删除等操作
"""
import oss2
from typing import Optional
from common.config import (
    OSS_ENDPOINT,
    OSS_BUCKET_NAME,
    OSS_ACCESS_KEY_ID,
    OSS_ACCESS_KEY_SECRET
)


class OSSHandler:
    """OSS文件处理器类"""

    def __init__(
            self,
            endpoint: Optional[str] = None,
            bucket_name: Optional[str] = None,
            access_key_id: Optional[str] = None,
            access_key_secret: Optional[str] = None
    ):
        """
        初始化OSS处理器

        Args:
            endpoint: OSS访问端点，如果不提供则使用配置中的值
            bucket_name: OSS Bucket名称，如果不提供则使用配置中的值
            access_key_id: AccessKey ID，如果不提供则使用配置中的值
            access_key_secret: AccessKey Secret，如果不提供则使用配置中的值
        """
        self.endpoint = endpoint or OSS_ENDPOINT
        self.bucket_name = bucket_name or OSS_BUCKET_NAME
        self.access_key_id = access_key_id or OSS_ACCESS_KEY_ID
        self.access_key_secret = access_key_secret or OSS_ACCESS_KEY_SECRET

        self.auth = None
        self.bucket = None

        # 自动连接
        self.connect()

    def connect(self) -> bool:
        """
        建立OSS连接

        Returns:
            连接是否成功
        """
        try:
            # 创建认证对象
            self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)

            # 创建Bucket对象
            self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

            # 测试连接 - 尝试列出bucket信息
            self.bucket.get_bucket_info()

            print(f"✓ 成功连接到OSS Bucket: {self.bucket_name}")
            return True

        except Exception as e:
            print(f"✗ 连接OSS时出错: {e}")
            self.auth = None
            self.bucket = None
            return False

    def upload_file(self, local_path: str, oss_path: str) -> Optional[str]:
        """
        上传本地文件到OSS

        Args:
            local_path: 本地文件路径
            oss_path: OSS中的文件路径

        Returns:
            上传成功返回文件的URL，失败返回None

        Example:
            >>> oss_handler = OSSHandler()
            >>> url = oss_handler.upload_file(
            ...     local_path="/tmp/data.xlsx",
            ...     oss_path="research_data/user123/data.xlsx"
            ... )
            >>> print(url)
            https://bucket-name.oss-cn-hangzhou.aliyuncs.com/research_data/user123/data.xlsx
        """
        if not self.bucket:
            print("OSS未连接，无法上传文件")
            return None

        try:
            # 上传文件
            self.bucket.put_object_from_file(oss_path, local_path)

            # 生成文件URL
            url = f"https://{self.bucket_name}.{self.endpoint.replace('https://', '').replace('http://', '')}/{oss_path}"

            print(f"✓ 文件上传成功: {url}")
            return url

        except Exception as e:
            print(f"✗ 上传文件时出错: {e}")
            return None

    def upload_bytes(self, data: bytes, oss_path: str) -> Optional[str]:
        """
        上传字节数据到OSS

        Args:
            data: 字节数据
            oss_path: OSS中的文件路径

        Returns:
            上传成功返回文件的URL，失败返回None

        Example:
            >>> oss_handler = OSSHandler()
            >>> with open("data.xlsx", "rb") as f:
            ...     data = f.read()
            >>> url = oss_handler.upload_bytes(
            ...     data=data,
            ...     oss_path="research_data/user123/data.xlsx"
            ... )
        """
        if not self.bucket:
            print("OSS未连接，无法上传数据")
            return None

        try:
            # 上传字节数据
            self.bucket.put_object(oss_path, data)

            # 生成文件URL
            url = f"https://{self.bucket_name}.{self.endpoint.replace('https://', '').replace('http://', '')}/{oss_path}"

            print(f"✓ 数据上传成功: {url}")
            return url

        except Exception as e:
            print(f"✗ 上传数据时出错: {e}")
            return None

    def download_file(self, oss_path: str, local_path: str) -> bool:
        """
        从OSS下载文件到本地

        Args:
            oss_path: OSS中的文件路径
            local_path: 本地保存路径

        Returns:
            下载是否成功

        Example:
            >>> oss_handler = OSSHandler()
            >>> success = oss_handler.download_file(
            ...     oss_path="research_data/user123/data.xlsx",
            ...     local_path="/tmp/downloaded_data.xlsx"
            ... )
        """
        if not self.bucket:
            print("OSS未连接，无法下载文件")
            return False

        try:
            # 下载文件
            self.bucket.get_object_to_file(oss_path, local_path)

            print(f"✓ 文件下载成功: {local_path}")
            return True

        except Exception as e:
            print(f"✗ 下载文件时出错: {e}")
            return False

    def download_bytes(self, oss_path: str) -> Optional[bytes]:
        """
        从OSS下载文件内容为字节数据

        Args:
            oss_path: OSS中的文件路径

        Returns:
            文件字节数据，失败返回None

        Example:
            >>> oss_handler = OSSHandler()
            >>> data = oss_handler.download_bytes("research_data/user123/data.xlsx")
            >>> if data:
            ...     with open("local_file.xlsx", "wb") as f:
            ...         f.write(data)
        """
        if not self.bucket:
            print("OSS未连接，无法下载数据")
            return None

        try:
            # 下载文件内容
            result = self.bucket.get_object(oss_path)
            data = result.read()

            print(f"✓ 数据下载成功: {len(data)} 字节")
            return data

        except Exception as e:
            print(f"✗ 下载数据时出错: {e}")
            return None

    def delete_file(self, oss_path: str) -> bool:
        """
        删除OSS中的文件

        Args:
            oss_path: OSS中的文件路径

        Returns:
            删除是否成功

        Example:
            >>> oss_handler = OSSHandler()
            >>> success = oss_handler.delete_file("research_data/user123/data.xlsx")
        """
        if not self.bucket:
            print("OSS未连接，无法删除文件")
            return False

        try:
            # 删除文件
            self.bucket.delete_object(oss_path)

            print(f"✓ 文件删除成功: {oss_path}")
            return True

        except Exception as e:
            print(f"✗ 删除文件时出错: {e}")
            return False

    def file_exists(self, oss_path: str) -> bool:
        """
        检查OSS中的文件是否存在

        Args:
            oss_path: OSS中的文件路径

        Returns:
            文件是否存在

        Example:
            >>> oss_handler = OSSHandler()
            >>> if oss_handler.file_exists("research_data/user123/data.xlsx"):
            ...     print("文件存在")
        """
        if not self.bucket:
            print("OSS未连接，无法检查文件")
            return False

        try:
            # 检查文件是否存在
            exists = self.bucket.object_exists(oss_path)
            return exists

        except Exception as e:
            print(f"✗ 检查文件时出错: {e}")
            return False

    def get_file_url(self, oss_path: str, expires: int = 3600) -> Optional[str]:
        """
        生成文件的临时访问URL（带签名）

        Args:
            oss_path: OSS中的文件路径
            expires: URL过期时间（秒），默认3600秒（1小时）

        Returns:
            临时访问URL，失败返回None

        Example:
            >>> oss_handler = OSSHandler()
            >>> url = oss_handler.get_file_url(
            ...     oss_path="research_data/user123/data.xlsx",
            ...     expires=7200  # 2小时
            ... )
            >>> print(url)
            https://bucket.oss-cn-hangzhou.aliyuncs.com/research_data/user123/data.xlsx?Expires=...
        """
        if not self.bucket:
            print("OSS未连接，无法生成URL")
            return None

        try:
            # 生成签名URL
            url = self.bucket.sign_url('GET', oss_path, expires)

            print(f"✓ 生成临时URL成功，有效期: {expires}秒")
            return url

        except Exception as e:
            print(f"✗ 生成URL时出错: {e}")
            return None

    def list_files(self, prefix: str = "", max_keys: int = 100) -> list:
        """
        列出OSS中的文件

        Args:
            prefix: 文件路径前缀，用于过滤
            max_keys: 最多返回的文件数量

        Returns:
            文件信息列表，每个元素包含key、size、last_modified等

        Example:
            >>> oss_handler = OSSHandler()
            >>> files = oss_handler.list_files(
            ...     prefix="research_data/user123/",
            ...     max_keys=10
            ... )
            >>> for file in files:
            ...     print(f"{file['key']}: {file['size']} bytes")
        """
        if not self.bucket:
            print("OSS未连接，无法列出文件")
            return []

        try:
            # 列出文件
            result = []
            for obj in oss2.ObjectIteratorV2(self.bucket, prefix=prefix, max_keys=max_keys):
                result.append({
                    "key": obj.key,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag
                })

            print(f"✓ 列出文件成功，共 {len(result)} 个文件")
            return result

        except Exception as e:
            print(f"✗ 列出文件时出错: {e}")
            return []

    def get_file_info(self, oss_path: str) -> Optional[dict]:
        """
        获取OSS中文件的详细信息

        Args:
            oss_path: OSS中的文件路径

        Returns:
            文件信息字典，包含size、content_type、last_modified等

        Example:
            >>> oss_handler = OSSHandler()
            >>> info = oss_handler.get_file_info("research_data/user123/data.xlsx")
            >>> if info:
            ...     print(f"文件大小: {info['size']} bytes")
            ...     print(f"最后修改: {info['last_modified']}")
        """
        if not self.bucket:
            print("OSS未连接，无法获取文件信息")
            return None

        try:
            # 获取文件元信息
            meta = self.bucket.get_object_meta(oss_path)

            info = {
                "size": meta.headers.get('Content-Length'),
                "content_type": meta.headers.get('Content-Type'),
                "last_modified": meta.headers.get('Last-Modified'),
                "etag": meta.headers.get('ETag')
            }

            print(f"✓ 获取文件信息成功: {oss_path}")
            return info

        except Exception as e:
            print(f"✗ 获取文件信息时出错: {e}")
            return None