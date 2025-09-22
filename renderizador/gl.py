#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    @staticmethod
    def order_winding(points):
        """
        A função order_winding ordena os vértices do triângulo
        no sentido anti-horário (winding order).

        Isso é feito calculando o centróide (média das coordenadas
        dos vértices) e obtendo o ângulo de cada vértice em relação
        a esse centróide. Esses ângulos são usados para ordenar os 
        vértices de forma decrescente, pois, diferentemente do círculo
        trigonométrico convencional (com eixo y para cima), no sistema
        de coordenadas de píxeis o eixo y cresce para baixo, o que 
        inverte o sentido do círculo trigonométrico.
        """

        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        ordered = sorted(points, key=(lambda p: math.atan2(p[1] - cy, p[0] - cx)), reverse=True)

        return ordered

    @staticmethod
    def is_inside(points, p):
        """
        A função is_inside analisa se o ponto está em cada um dos semiplanos das arestas do triângulo,
        e desenha o pixel caso este seja o caso.
        """
        def L(p1, p2, p):
            """
            A função L retorna se o ponto p -> (x, y) está no semiplano de uma das arestas do triângulo,
            usando a função:
            L(x, y) = (y1 - y0)x - (x1 - x0)y + y0(x1 - x0) - x0(y1 - y0)
            e retornando um valor booleano caso o resultado da função seja maior ou igual a zero,
            dado o fato que usamos o sistema de coordenadas de pixels, onde o eixo y é orientado
            para baixo, ao invés de para cima.
            """
            result = (p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p1[1] * (p2[0] - p1[0]) - p1[0] * (p2[1] - p1[1])
            return result >= 0

        v1, v2, v3 = L(points[0], points[1], p), L(points[1], points[2], p), L(points[2], points[0], p)
        if v1 and v2 and v3:
            return True

    @staticmethod
    def pushmatrix(m):
       GL.transformation_stack.append(m)

    @staticmethod
    def popmatrix():
        return GL.transformation_stack.pop()
    
    @staticmethod
    def translation_matrix(xt, yt, zt):
        """
        A função translation_matrix gera a matriz
        de translação homogênea a partir dos valores
        de translação para x, y e z. 
        """
        return np.array([
            [1, 0, 0, xt],
            [0, 1, 0, yt],
            [0, 0, 1, zt],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def scale_matrix(xs, ys, zs):
        """
        A função scale_matrix gera a matriz de
        escala homogênea a partir dos valores de 
        escala para x, y e z.
        """
        return np.array([
            [xs, 0, 0, 0],
            [0, ys, 0, 0],
            [0, 0, zs, 0],
            [0, 0, 0, 1]
        ])      
    
    @staticmethod
    def quaternion_rotation_matrix(x: float, y: float, z: float, t: float):
        """
        A função quaternion_rotation_matrix usa 
        eixo da rotação (x, y, z), em conjunto 
        com o valor em radianos para theta
        para criar a matriz de rotação.
        """

        def generate_quaternion(x: float, y: float, z: float, theta: float):
            """
            A função generate_quaternion usa o 
            eixo da rotação (x, y, z) em conjunto
            com o valor em radianos para theta
            para criar o quatérnio que irá compor a 
            matriz de rotação.

            um quatérnio [qi, qj, qk, qr] é, na
            prática, o vetor [
                ux*sin(theta/2),
                uy*sin(theta/2),
                uz*sin(theta/2),
                cos(theta/2)
            ]

            sendo ux, uy e uz os versores do eixo
            (vetor [x, y, z] divido pela norma). 
            """
            vector = np.array([x, y, z])
            norm = np.linalg.norm(vector)

            # se a norma do vetor é zero, não
            # há rotação, e o se usa o quatérnio 1,
            # equivalente a identidade.
            if norm == 0:
                return (0, 0, 0, 1)
            
            normalized_vector = vector / norm
            ux, uy, uz = normalized_vector

            return (
                ux * math.sin(theta/2),
                uy * math.sin(theta/2),
                uz * math.sin(theta/2),
                math.cos(theta/2)
            )
        
        qi, qj, qk, qr = generate_quaternion(x, y, z, t)
        
        return np.array([
            [1 - 2*(qj**2 + qk**2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr), 0],
            [2*(qi*qj + qk*qr), 1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr), 0],
            [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1 - 2*(qi**2 + qj**2), 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def look_at_matrix(up, at, eye):
        """
        A função look_at_matrix utiliza as informações
        de posição e orientação da câmera para criar 
        a matriz de look_at, que transfomra as coordenadas
        do mundo para o espaço da câmera, permitindo a
        manipulação de objetos a partir da perspectiva da
        câmera.
        """
        w = at - eye
        w = w / float(np.linalg.norm(w))

        u = np.linalg.cross(w, up)
        u = u / float(np.linalg.norm(u))

        v = np.linalg.cross(u, w)
        v = v / float(np.linalg.norm(v))

        R = np.array([
            [u[0], v[0], -w[0], 0],
            [u[1], v[1], -w[1], 0],
            [u[2], v[2], -w[2], 0],
            [0, 0, 0, 1]
        ]).T

        E = np.array([
            [1, 0, 0, -eye[0]],
            [0, 1, 0, -eye[1]],
            [0, 0, 1, -eye[2]],
            [0, 0, 0, 1]
        ])

        return R @ E
    
    @staticmethod
    def perspective_projection_matrix(far, near, right, top):
        """
        A função perspective_projection_matrix cria
        a matriz de projeção, ou perspectiva, que projeta
        o espaço 3D em 2D, mantendo a perspectiva correta 
        esperada, ao invés de simplesmente planificar os objetos,
        como uma sombra faria, por exemplo.
        """
        return np.array([
            [near/right, 0, 0, 0],
            [0, near/top, 0, 0],
            [0, 0, -((far+near)/(far-near)), -(2*far*near)/(far-near)],
            [0, 0, -1, 0]
        ])
    
    @staticmethod
    def screen_transformation_matrix(W,H):
        """
        A função screen_transfomation_matrix  
        cria a matriz de transformação para a tela,
        responsável por mapear os pontos no espaço de 
        coordenadas normalizado da câmera para o espaço 
        de coordenada de pixels.
        """
        return np.array([
            [W/2, 0, 0, W/2],
            [0, -H/2, 0, H/2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def calculate_baricentric_coordinates(A, B, C, point):
        alpha = ((-(point[0] - B[0]) * (C[1] - B[1]) + (point[1] - B[1]) * (C[0] - B[0])) /
                (-(A[0] - B[0]) * (C[1] - B[1]) + (A[1] - B[1]) * (C[0] - B[0])))
        
        beta = ((-(point[0] - C[0]) * (A[1] - C[1]) + (point[1] - C[1]) * (A[0] - C[0])) /
                (-(B[0] - C[0]) * (A[1] - C[1]) + (B[1] - C[1]) * (A[0] - C[0])))
        
        gamma = 1 - alpha - beta 

        return round(alpha, 3), round(beta, 3), round(gamma, 3)

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

        GL.transformation_stack = [np.identity(4)]
        GL.view_matrix = np.identity(4)
        GL.vision_perspective = []
        GL.perspective_matrix = np.identity(4)
        GL.screen_matrix = np.identity(4)

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        # Itera de dois em dois valores pelos pontos
        for i in range(0, len(point), 2):
            # Formata os valores de x e y para o formato da função draw_pixel
            cur_point = [int(point[i]), int(point[i+1])]

            # Formata as cores para o formato do frame_buffer
            color = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]

            # E finalmente, desenha o pixel no Framebuffer se ele se encontra na tela
            if cur_point[0] >= 0 and cur_point[0] < GL.width and cur_point[1] >= 0 and cur_point[1] < GL.height:
                gpu.GPU.draw_pixel(cur_point, gpu.GPU.RGB8, color)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        def draw_line(pointA, pointB, color):
            """
            A funçao draw_line desenha uma linha entre dois pontos quaisquer.
            Isso é feito calculando os deltas entre as coordenadas dos pontos,
            e então escolhendo qual desses deltas é maior (valor absoluto), ou seja, em que eixo 
            a distância a percorrer desenhando será maior, chamado de step.

            Dado o step, se percorre a distância e, para cada passo, somam se os steps individuais, deltaX ou Y / step, 
            que será um valor entre 0 e 1, à coordenada atual para x e y, e então um pixel é desenhado
            na coordenada resultante, se essa coordenada se encontra no quadro (frame buffer).
            """
            deltaX = pointB[0] - pointA[0]
            deltaY = pointB[1] - pointA[1]

            step = max(abs(deltaX), abs(deltaY))

            stepX = deltaX / step
            stepY = deltaY / step

            for i in range(round(step+1)):
                cur_x = pointA[0] + round(i * stepX)
                cur_y = pointA[1] + round(i * stepY)

                if cur_x >= 0 and cur_x < GL.width and cur_y >= 0 and cur_y < GL.height:
                    gpu.GPU.draw_pixel([cur_x, cur_y], gpu.GPU.RGB8, color)

        for i in range(0, len(lineSegments) - 2, 2):
            pointA = (int(lineSegments[i]), int(lineSegments[i + 1]))
            pointB = (int(lineSegments[i + 2]), int(lineSegments[i + 3]))

            # Formata as cores para o formato do frame_buffer
            color = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]
            draw_line(pointA, pointB, color)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        # Extrai a cor
        color = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]

        # Define uma tolerância, dado que, para uma tela de pixels,
        # é impossível ou muito raro que a distância euclidiana dos 
        # pontos que formaram a circunferência seja exata ao
        # calcular (x**2 + y**2) ** 0.5.
        tolerance = 0.5 

        # A ideia desse algoritmo é iterar pelo quadrado 
        # formado pelo raio, e toda vez que a distância
        # euclidiana calculada pros pontos x e y da origem
        # for menor do que a tolerância (o correto seria ser zero
        # mas dado a tela de pixels, precisamos de uma tolerância)
        # pintamos o pixel.
        for x in range(round(radius)+1): 
            for y in range(round(radius)+1): 
                
                radial_distance = math.sqrt(y**2 + x**2) 

                if abs(radial_distance - radius) <= tolerance: 
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        COLOR = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]

        for i in range(0, len(vertices), 6):
            p1 = (vertices[i], vertices[i+1])
            p2 = (vertices[i+2], vertices[i+3])
            p3 = (vertices[i+4], vertices[i+5])

            winding_ordered_points = GL.order_winding([p1, p2, p3])

            min_x, max_x = int(min([p1[0], p2[0], p3[0]]) - 1), int(max([p1[0], p2[0], p3[0]]) + 1)
            min_y, max_y = int(min([p1[1], p2[1], p3[1]]) - 1), int(max([p1[1], p2[1], p3[1]]) + 1)

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if 0 <= x < GL.width and 0 <= y < GL.height and GL.is_inside(winding_ordered_points, (x, y)):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, COLOR)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        COLOR = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]

        for i in range(0, len(point), 9):
            triangle = np.array([
                [point[i], point[i+1], point[i+2], 1],  
                [point[i+3], point[i+4], point[i+5], 1],  
                [point[i+6], point[i+7], point[i+8], 1]
            ]).T

            # Aplica-se as matrizes de world, view e perspective, todas homogêneas
            t_matrix = GL.perspective_matrix @ GL.view_matrix @ GL.transformation_stack[-1]

            triangle = t_matrix @ triangle

            # Como agora temos termos diferente de zero no quarto componente, é
            # necessário fazer a Divisão Homogênea para normalizar esse componente
            # novamente.
            triangle[0, :] = triangle[0, :] / triangle[3, :]
            triangle[1, :] = triangle[1, :] / triangle[3, :]
            triangle[2, :] = triangle[2, :] / triangle[3, :]
            triangle[3, :] = triangle[3, :] / triangle[3, :]

            # Aplicamos a matriz de transformação para a tela, após o Homogeneous Divide
            triangle = GL.screen_transformation_matrix(GL.width, GL.height) @ triangle

            # Extraimos os vértices do triângulo em 2D e ordenamos para
            # realizar a checagem se os pontos estão dentro do plano do triângulo
            p1 = (triangle[0][0], triangle[1][0])
            p2 = (triangle[0][1], triangle[1][1])
            p3 = (triangle[0][2], triangle[1][2])
            winding_ordered_points = GL.order_winding([p1, p2, p3])

            min_x, max_x = int(min([p1[0], p2[0], p3[0]]) - 1), int(max([p1[0], p2[0], p3[0]]) + 1)
            min_y, max_y = int(min([p1[1], p2[1], p3[1]]) - 1), int(max([p1[1], p2[1], p3[1]]) + 1)

            # E finalmente, desenhamos o triângulo
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if 0 <= x < GL.width and 0 <= y < GL.height and GL.is_inside(winding_ordered_points, (x, y)):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, COLOR)
            

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O primeiro passo para criar a matriz view é criar uma 
        # matriz de rotação com base na orientação da câmera.
        # Isso permite obter as direções up e forward para a 
        # orientação da câmera, a partir dos versores dos
        # eixos y e z (negativo)
        rotation_matrix = GL.quaternion_rotation_matrix(
            orientation[0], 
            orientation[1], 
            orientation[2], 
            orientation[3]
        )
        rotation_matrix = rotation_matrix[:3, :3]

        base_forward = np.array([0, 0, -1])
        base_up = np.array([0, 1, 0])

        camera_forward = rotation_matrix @ base_forward

        camera_up = rotation_matrix @ base_up

        # O valor de eye é a posição da câmera no espaço,
        # enquanto o valor de at é a posição do objeto para o qual
        # a câmera está apontando em relação à posição da câmera
        eye = np.array(position)
        at = eye + camera_forward
    
        # Com essas informações e a informação do fov podemos
        # construir duas matrizes: A matriz de look at e a 
        # matriz de perspectiva.
        view_matrix = GL.look_at_matrix(up=camera_up, at=at, eye=eye)
        GL.view_matrix = view_matrix

        # A matriz de perspectiva projeta os pontos 3D em 2D,
        # permitindo representar a perspectiva direta de profundidade
        # no nosso plano da tela.
        top = GL.near * np.tan(fieldOfView/2)
        bottom = -top
        right = top * GL.width / GL.height
        left = -right

        GL.vision_perspective = [
            top,
            bottom,
            right,
            left
        ]
        GL.perspective_matrix = GL.perspective_projection_matrix(GL.far, GL.near, right, top)

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.
        
        t_matrix = GL.translation_matrix(translation[0], translation[1], translation[2])
        r_matrix = GL.quaternion_rotation_matrix(rotation[0], rotation[1], rotation[2], rotation[3])
        s_matrix = GL.scale_matrix(scale[0], scale[1], scale[2])

        transformation_matrix = t_matrix @ r_matrix @ s_matrix

        # Finalmente, após obter a matriz combinada de transformação,
        # empilhamos pra manter a hierarquia de transformações
        GL.transformation_stack.append(GL.transformation_stack[-1] @ transformation_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        GL.popmatrix()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        COLOR = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]

        # Única alteração feita sobre o código de triangle set
        # é que agora nós percorremos de 3 em 3, ao invés de 9
        # em 9, para reutilizar os vértices que conectam os triângulos
        # da tira.
        for i in range(0, len(point)-6, 3):
            triangle = np.array([
                [point[i], point[i+1], point[i+2], 1],  
                [point[i+3], point[i+4], point[i+5], 1],  
                [point[i+6], point[i+7], point[i+8], 1]
            ]).T

            # Aplica-se as matrizes de world, view e perspective, todas homogêneas
            t_matrix = GL.perspective_matrix @ GL.view_matrix @ GL.transformation_stack[-1]

            triangle = t_matrix @ triangle

            # Como agora temos termos diferente de zero no quarto componente, é
            # necessário fazer a Divisão Homogênea para normalizar esse componente
            # novamente.
            triangle[0, :] = triangle[0, :] / triangle[3, :]
            triangle[1, :] = triangle[1, :] / triangle[3, :]
            triangle[2, :] = triangle[2, :] / triangle[3, :]
            triangle[3, :] = triangle[3, :] / triangle[3, :]

            # Aplicamos a matriz de transformação para a tela, após o Homogeneous Divide
            triangle = GL.screen_transformation_matrix(GL.width, GL.height) @ triangle

            # Extraimos os vértices do triângulo em 2D e ordenamos para
            # realizar a checagem se os pontos estão dentro do plano do triângulo
            p1 = (triangle[0][0], triangle[1][0])
            p2 = (triangle[0][1], triangle[1][1])
            p3 = (triangle[0][2], triangle[1][2])
            winding_ordered_points = GL.order_winding([p1, p2, p3])

            min_x, max_x = int(min([p1[0], p2[0], p3[0]]) - 1), int(max([p1[0], p2[0], p3[0]]) + 1)
            min_y, max_y = int(min([p1[1], p2[1], p3[1]]) - 1), int(max([p1[1], p2[1], p3[1]]) + 1)

            # E finalmente, desenhamos o triângulo
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if 0 <= x < GL.width and 0 <= y < GL.height and GL.is_inside(winding_ordered_points, (x, y)):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, COLOR)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        COLOR = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]

        # Para essa implementação, é possível construir uma lista de pontos, 
        # e portanto, acessar/armazenar cada vértice apenas uma vez, devido
        # aos ponteiros de cada triângulo aos indíces de seus vértices.
        vertices = []
        for i in range(0, len(point), 3):
            vertices.append((point[i], point[i+1], point[i+2], 1))

        for i in range(len(index)-3):
            triangle = np.array([
                vertices[index[i]],
                vertices[index[i+1]],
                vertices[index[i+2]]
            ]).T

            # Aplica-se as matrizes de world, view e perspective, todas homogêneas
            t_matrix = GL.perspective_matrix @ GL.view_matrix @ GL.transformation_stack[-1]

            triangle = t_matrix @ triangle

            # Como agora temos termos diferente de zero no quarto componente, é
            # necessário fazer a Divisão Homogênea para normalizar esse componente
            # novamente.
            triangle[0, :] = triangle[0, :] / triangle[3, :]
            triangle[1, :] = triangle[1, :] / triangle[3, :]
            triangle[2, :] = triangle[2, :] / triangle[3, :]
            triangle[3, :] = triangle[3, :] / triangle[3, :]

            # Aplicamos a matriz de transformação para a tela, após o Homogeneous Divide
            triangle = GL.screen_transformation_matrix(GL.width, GL.height) @ triangle

            # Extraimos os vértices do triângulo em 2D e ordenamos para
            # realizar a checagem se os pontos estão dentro do plano do triângulo
            p1 = (triangle[0][0], triangle[1][0])
            p2 = (triangle[0][1], triangle[1][1])
            p3 = (triangle[0][2], triangle[1][2])
            winding_ordered_points = GL.order_winding([p1, p2, p3])

            min_x, max_x = int(min([p1[0], p2[0], p3[0]]) - 1), int(max([p1[0], p2[0], p3[0]]) + 1)
            min_y, max_y = int(min([p1[1], p2[1], p3[1]]) - 1), int(max([p1[1], p2[1], p3[1]]) + 1)

            # E finalmente, desenhamos o triângulo
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if 0 <= x < GL.width and 0 <= y < GL.height and GL.is_inside(winding_ordered_points, (x, y)):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, COLOR)
            

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        if not colorPerVertex or not color or not colorIndex:
            colorPerVertex = False
            COLOR = [int(255 * colors['emissiveColor'][i]) for i in range(len(colors["emissiveColor"]))]

        # Criamos a lista de vértices
        vertices = []
        for i in range(0, len(coord), 3):
            vertices.append((coord[i], coord[i+1], coord[i+2], 1))
        
        colors = None
        if colorPerVertex:
            colors = []
            for i in range(0, len(color), 3):
                colors.append(np.asarray([[color[i],color[i+1],color[i+2]]]))

        hasTexture = False
        if len(current_texture) != 0:
            hasTexture = True
            texture = gpu.GPU.load_texture(current_texture[0])
            textures = []
            for i in range(0, len(texCoord), 2):
                textures.append([round(texCoord[i]), round(texCoord[i+1])])

        # Para essa implementação, existem 'sets' de triângulos,
        # onde para cada set como [0, 1, 2, 3, -1] todos os triângulos
        # compartilham do vértice 0, sendo que o valor -1 sinaliza o fim
        # de um set, alterando o vértice compartilhado, como em
        # [0, 1, 2, 3, -1, 5, 6, 7, 8, -1], onde os triângulos seriam
        # (0, 1, 2), (0, 2, 3), (5, 6, 7) e (5, 6, 8).

        # Para isso, mantenho a referência do vértice origem, e verifico
        # se em algum momento cheguei ao -1. Caso sim, redefino a origem,
        # ou finalizo a execução, no caso do fim do vetor de vértices.
        origin = 0
        i = 1
        while i < len(coordIndex) - 2:

            if coordIndex[i] == -1:
                if i+1 < len(coordIndex):
                    origin = i+1
                    i += 2
                    continue

            if i+1 < len(coordIndex) and coordIndex[i+1] == -1:
                if i+2 < len(coordIndex):
                    origin = i+2
                    i += 3
                    continue

            triangle = np.array([
                vertices[coordIndex[origin]],
                vertices[coordIndex[i]],
                vertices[coordIndex[i+1]]
            ]).T

            if colorPerVertex:
                point_colors = [colors[colorIndex[origin]], colors[colorIndex[i]], colors[colorIndex[i+1]]]

            if hasTexture:
                point_texture_uv = [textures[texCoordIndex[origin]], textures[texCoordIndex[i]], textures[texCoordIndex[i+1]]]

            # Aplica-se as matrizes de world, view e perspective, todas homogêneas
            t_matrix = GL.perspective_matrix @ GL.view_matrix @ GL.transformation_stack[-1]

            triangle = t_matrix @ triangle

            # Precisamos extrair o z de cada vértice antes de fazer a Divisão Homogênea,
            # pois para realizar a média harmônica precisamos do Z do espaço da câmera
            vertexZs = (triangle[2][0], triangle[2][1], triangle[2][2])

            # Como agora temos termos diferente de zero no quarto componente, é
            # necessário fazer a Divisão Homogênea para normalizar esse componente
            # novamente.
            triangle[0, :] = triangle[0, :] / triangle[3, :]
            triangle[1, :] = triangle[1, :] / triangle[3, :]
            triangle[2, :] = triangle[2, :] / triangle[3, :]
            triangle[3, :] = triangle[3, :] / triangle[3, :]

            # Aplicamos a matriz de transformação para a tela, após o Homogeneous Divide
            triangle = GL.screen_transformation_matrix(GL.width, GL.height) @ triangle

            # Extraimos os vértices do triângulo em 2D e ordenamos para
            # realizar a checagem se os pontos estão dentro do plano do triângulo
            p1 = (triangle[0][0], triangle[1][0], triangle[2][0])
            p2 = (triangle[0][1], triangle[1][1], triangle[2][1])
            p3 = (triangle[0][2], triangle[1][2], triangle[2][2])
            winding_ordered_points = GL.order_winding([p1, p2, p3])

            min_x, max_x = int(min([p1[0], p2[0], p3[0]]) - 1), int(max([p1[0], p2[0], p3[0]]) + 1)
            min_y, max_y = int(min([p1[1], p2[1], p3[1]]) - 1), int(max([p1[1], p2[1], p3[1]]) + 1)

            # E finalmente, desenhamos o triângulo
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if 0 <= x < GL.width and 0 <= y < GL.height and GL.is_inside(winding_ordered_points, (x, y)):
                        if colorPerVertex or hasTexture:
                            alpha, beta, gamma = GL.calculate_baricentric_coordinates(p1, p2, p3, (x, y))

                            # Construimos o Z projetado do ponto amostrado, usando a média harmônica ponderada
                            # das coordenadas baricêntricas 
                            Z = 1 / (alpha * 1/vertexZs[0] + beta * 1/vertexZs[1] + gamma * 1/vertexZs[2]) 

                            # E usamos a divisão pelo Z dos vértices e então uma multiplicação pelo Z do ponto amostrado
                            # transformando a interpolação numa transformação afim
                            
                            # Portanto, o valor de cor/textura será o valor interpolado entre os valores
                            # dos vértices, usado o cálculo baricêntrico.

                            if colorPerVertex:
                                COLOR = Z * (alpha * point_colors[0][:] / vertexZs[0] + beta * point_colors[1][:] / vertexZs[1] + gamma * point_colors[2][:] / vertexZs[2])
                                COLOR = [round(255*i) for i in COLOR[0]]
                            if hasTexture:
                                h, w = texture.shape[:2]
                                h, w = h-1, w-1

                                u = (
                                    alpha*(point_texture_uv[0][0]/vertexZs[0]) 
                                    + beta*(point_texture_uv[1][0]/vertexZs[1]) 
                                    + gamma*(point_texture_uv[2][0]/vertexZs[2])
                                ) * Z

                                v = (
                                    alpha*(point_texture_uv[0][1]/vertexZs[0])
                                    + beta*(point_texture_uv[1][1]/vertexZs[1])
                                    + gamma*(point_texture_uv[2][1]/vertexZs[2])
                                ) * Z

                                u = round(u*w) if u*w < w else w 
                                v = -round(v*h) if v*h < h else h
                                COLOR = texture[u][v][:3]

                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, COLOR)

            i += 1

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
